"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
import torch
from torch.autograd import Function
import pickle
from monai.transforms import MapTransform


def pickle_read(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_write(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=(3, 3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(
        self, in_channels, res_channels=0, last_layer=False, num_classes=None
    ) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
            last_layer == True and num_classes != None
        ), "Invalid arguments"
        self.upconv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(2, 2, 2),
            stride=2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels + res_channels,
            out_channels=in_channels // 2,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_channels // 2,
            out_channels=in_channels // 2,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(
                in_channels=in_channels // 2,
                out_channels=num_classes,
                kernel_size=(1, 1, 1),
            )

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual != None:
            out = torch.cat((out, residual), 1)
        middle_features = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(middle_features)))
        if self.last_layer:
            out = self.conv3(out)
        return out, middle_features


class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        level_channels=[64, 128, 256],
        bottleneck_channel=512,
    ) -> None:
        super(UNet3D, self).__init__()
        # level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.down_convs.append(
            Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        )
        for in_channels, out_channels in zip(level_channels[:-1], level_channels[1:]):
            self.down_convs.append(
                Conv3DBlock(in_channels=in_channels, out_channels=out_channels)
            )

        # self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        # self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        # self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(
            in_channels=level_channels[-1],
            out_channels=bottleneck_channel,
            bottleneck=True,
        )

        self.up_convs.append(
            UpConv3DBlock(
                in_channels=bottleneck_channel, res_channels=level_channels[-1]
            )
        )
        for i, in_channels, out_channels in zip(
            range(len(level_channels) - 1),
            level_channels[-1:0:-1],
            level_channels[-2::-1],
        ):
            if i < len(level_channels) - 2:
                self.up_convs.append(
                    UpConv3DBlock(in_channels=in_channels, res_channels=out_channels)
                )
            else:
                self.up_convs.append(
                    UpConv3DBlock(
                        in_channels=in_channels,
                        res_channels=out_channels,
                        num_classes=num_classes,
                        last_layer=True,
                    )
                )

        # self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        # self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        # self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    def freeze_decoder_layers(self):
        self.s_block3.requires_grad = False
        self.s_block2.requires_grad = False
        self.s_block1.requires_grad = False

    def forward(self, x):
        residuals = []
        for down_conv in self.down_convs:
            x, res = down_conv(x)
            residuals.append(res)

        x, _ = self.bottleNeck(x)

        for i, up_conv in enumerate(self.up_convs):
            res = residuals[-i - 1]
            x, _ = up_conv(x, res)

        return x, residuals

        # Analysis path forward feed
        # out, residual_level1 = self.a_block1(input) #(128,128,16,in_channel=1) -> (64,64,8,64)
        # out, residual_level2 = self.a_block2(out) # -> (64,64,8,64) -> (32,32,4,128)
        # out, residual_level3 = self.a_block3(out) # -> (32,32,4,128) -> (16,16,2,256)
        # lowest, _ = self.bottleNeck(out) # (16,16,2,256) -> (8,8,1,512)

        # #Synthesis path forward feed
        # up1,_ = self.s_block3(lowest, residual_level3) # (8,8,1,512) -> (32,32,2,256)
        # up2,_ = self.s_block2(up1, residual_level2) # (32,32,2,256) -> (64,64,4,128)
        # out,_ = self.s_block1(up2, residual_level1) # (64,64,8,128) -> (128,128,num_classes)
        # return out, (residual_level1, residual_level2, residual_level3, lowest, up1, up2)

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=(3, 3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(
        self, in_channels, res_channels=0, last_layer=False, num_classes=None
    ) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
            last_layer == True and num_classes != None
        ), "Invalid arguments"
        self.upconv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(2, 2, 2),
            stride=2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels + res_channels,
            out_channels=in_channels // 2,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_channels // 2,
            out_channels=in_channels // 2,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(
                in_channels=in_channels // 2,
                out_channels=num_classes,
                kernel_size=(1, 1, 1),
            )

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual != None:
            out = torch.cat((out, residual), 1)
        middle_features = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(middle_features)))
        if self.last_layer:
            out = self.conv3(out)
        return out, middle_features



class OutputChannel3DConverter(nn.Module):
    def __init__(
        self, baseModel, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(OutputChannel3DConverter, self).__init__()

        self.baseModel = baseModel
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.LeakyReLU()  # .GELU()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.baseModel(x)
        x = self.relu(x)
        x = self.conv(x)

        return x


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


def concat_unet_features(reshaped_size_xy, reshaped_size_z, features):
    residual_level1, residual_level2, residual_level3, lowest = features
    batch_size = residual_level1.shape[0]

    residual_level1 = residual_level1.view(
        batch_size, -1, reshaped_size_xy, reshaped_size_xy, reshaped_size_z
    )
    residual_level2 = residual_level2.view(
        batch_size, -1, reshaped_size_xy, reshaped_size_xy, reshaped_size_z
    )
    residual_level3 = residual_level3.view(
        batch_size, -1, reshaped_size_xy, reshaped_size_xy, reshaped_size_z
    )
    lowest = lowest.view(
        batch_size, -1, reshaped_size_xy, reshaped_size_xy, reshaped_size_z
    )

    res = torch.concat(
        (residual_level1, residual_level2, residual_level3, lowest), dim=1
    )

    return res


class Critic(nn.Module):
    def __init__(self, in_channels, gr_coeff=1) -> None:
        super(Critic, self).__init__()

        # self.conv1 =  nn.Sequential(
        #     GradientReversal(gr_coeff),
        #     nn.Conv3d(in_channels, 64, (4, 4, 3), (2, 2, 1), (1, 1, 1), bias=True),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(64, 128, (4, 4, 3), (2, 2, 1), (1, 1, 1), bias=False),
        #     nn.BatchNorm3d(128),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.conv3 = nn.Sequential(
        #      nn.Conv3d(128, 256, (4, 4, 3), (2, 2, 1), (1, 1, 1), bias=False),
        #     nn.BatchNorm3d(256),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv3d(256, 512, (4, 4, 4), (2, 2, 2), (1, 1, 1), bias=False),
        #     nn.BatchNorm3d(512),
        #     nn.LeakyReLU(0.2, True),
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv3d(512, 1024, (4, 4, 4), (2, 2, 2), (1, 1, 1), bias=False),
        #     nn.BatchNorm3d(1024),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.conv6 = nn.Conv3d(1024,2048,4,1,0,bias=True)

        # self.fc = nn.Sequential(
        #     nn.Linear(2048,1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024,512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512,1),
        #     nn.Sigmoid()
        # )

        self.reversal = GradientReversal(gr_coeff)
        self.conv1 = Conv3DBlock(in_channels, 128)  # (32,32,4,256) -> (16,16,2,128)
        self.conv2 = Conv3DBlock(128, 256)  # (16,16,2,128) -> (8,8,1,256)
        self.conv3 = nn.Sequential(
            nn.Conv3d(256, 256, (3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
        )  # (8,8,1,256) -> (4,4,1,256)
        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # out = self.conv1(img)
        # out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)

        # out = torch.squeeze(out)

        # score = self.fc(out)

        # pdb.set_trace()

        img = self.reversal(img)
        out, _ = self.conv1(img)
        out, _ = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out


# ------------LOSS FUNCTIONS------------


class Dice(nn.Module):
    def __init__(self, num_classes):
        super(Dice, self).__init__()
        self.num_classes = num_classes

    def forward(self, predicted, target):
        predicted = torch.nn.functional.one_hot(
            predicted.long(), self.num_classes
        ).permute((0, 4, 1, 2, 3))
        one_hot_tar = torch.nn.functional.one_hot(
            target.long(), self.num_classes
        ).permute((0, 4, 1, 2, 3))

        pred_flat = predicted.view(predicted.size(0), self.num_classes, -1)
        tar_flat = one_hot_tar.view(one_hot_tar.size(0), self.num_classes, -1)

        dice_numinator = 2 * (pred_flat * tar_flat).sum(2)
        dice_denominator = pred_flat.sum(2) + tar_flat.sum(2)

        dice_denominator_zeros = dice_denominator == 0

        # in cases where both pred and tar have no area, we set the dice to 1
        dice_numinator[dice_denominator_zeros] = 1
        dice_denominator[dice_denominator_zeros] = 1

        dice = (dice_numinator / dice_denominator).mean(0)

        return dice


class DiceLoss(nn.Module):
    def __init__(self, num_classes, include_background=True, classes_to_include=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.include_background = include_background

        if classes_to_include is not None:
            self.classes_to_include = torch.ones(num_classes)

    def forward(self, predicted, target):
        predicted = torch.exp(F.log_softmax(predicted, dim=1))
        one_hot_tar = torch.nn.functional.one_hot(
            target.squeeze(1).long(), self.num_classes
        ).permute((0, 4, 1, 2, 3))

        pred_flat = predicted.view(predicted.size(0), self.num_classes, -1)
        tar_flat = one_hot_tar.view(one_hot_tar.size(0), self.num_classes, -1)

        dice_numinator = 2 * (pred_flat * tar_flat).sum(2)
        dice_denominator = torch.pow(pred_flat, 2).sum(2) + tar_flat.sum(2)

        dice = (dice_numinator / dice_denominator).mean(0)

        if self.include_background == False:
            dice = dice.as_tensor()[1:]

        return 1 - dice.mean()


# ------------Transforms------------


class NormalizeImageToRange(MapTransform):
    def __init__(self, keys, target_range):
        self.keys = keys
        self.target_range = target_range

    def __call__(self, data):
        image = data[self.keys[0]]

        # Map values from the range [a, b] to [c, d]
        a, b = image.min(), image.max()
        c, d = (
            self.target_range[0],
            self.target_range[1],
        )  # Replace with your desired range

        data[self.keys[0]] = (image - a) * ((d - c) / (b - a)) + c

        return data


class FilterClasses(MapTransform):
    def __init__(self, keys, classes_to_include):
        self.keys = keys
        self.classes_to_include = classes_to_include

    def __call__(self, data):
        labels = data[self.keys[1]]

        new_labels = torch.zeros_like(labels)

        classes_indexes = torch.nonzero(self.classes_to_include)[1]

        for index, class_num in enumerate(classes_indexes):
            new_labels[labels == class_num] = index

        data[self.keys[1]] = new_labels

        return data


class CropClass(MapTransform):
    def __init__(self, keys, class_number, amount_of_slices):
        super().__init__(keys)
        self.keys = keys
        self.class_number = class_number
        self.amount_of_slices = amount_of_slices

    def __call__(self, data):
        labels = data[self.keys[1]]

        class_bin_array = labels == self.class_number

        availabel_indexes = torch.any(torch.any(class_bin_array, dim=1), dim=1)[0]

        # pdb.set_trace()
        if availabel_indexes.any() == False:
            min = 0
            max = availabel_indexes.shape[0]
        else:
            min = torch.nonzero(availabel_indexes)[0].item()
            max = torch.nonzero(availabel_indexes)[-1].item()

        avg = round((max + min) / 2)
        min = avg - int(self.amount_of_slices / 2)
        max = avg + int(self.amount_of_slices / 2)

        if min < 0:
            min -= min
            max -= min

        for key in self.keys:
            data[key] = data[key][:, :, :, min:max]

        return data
