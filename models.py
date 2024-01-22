"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
import torch
from torch.autograd import Function


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

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)


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

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))


    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
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

    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        # level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.down_convs.append(Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0]))
        for in_channels, out_channels in zip(level_channels[:-1],level_channels[1:]):
          self.down_convs.append(Conv3DBlock(in_channels=in_channels, out_channels=out_channels))

        self.bottleNeck = Conv3DBlock(in_channels=level_channels[-1], out_channels=bottleneck_channel, bottleneck= True)


        self.up_convs.append(UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[-1]))
        for i,in_channels, out_channels in zip(range(len(level_channels)-1),level_channels[-1:0:-1],level_channels[-2::-1]):
          if i < len(level_channels)-2:
            self.up_convs.append(UpConv3DBlock(in_channels=in_channels, res_channels=out_channels))
          else:
            self.up_convs.append(UpConv3DBlock(in_channels=in_channels, res_channels=out_channels, num_classes = num_classes, last_layer = True))


    def freeze_decoder_layers(self):
      self.up_convs.requires_grad = False


    def forward(self, x):

        residuals = []
        for down_conv in self.down_convs:
          x,res = down_conv(x)
          residuals.append(res)

        x,_ = self.bottleNeck(x)

        for i,up_conv in enumerate(self.up_convs):
          res = residuals[-i-1]
          x,_ = up_conv(x, res)

        return x, residuals
    

# this model was implemented to allow for different number of ouput classes using a given base model
class OutputChannel3DConverter(nn.Module):
    def __init__(
        self, baseModel, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(OutputChannel3DConverter, self).__init__()

        self.baseModel = baseModel
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.LeakyReLU()
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
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply

class GradientReversal(nn.Module):
  def __init__(self, alpha):
      super().__init__()
      self.alpha = torch.tensor(alpha, requires_grad=False)

  def forward(self, x):
      return revgrad(x, self.alpha)    

class Critic(nn.Module):
    """
    This should be used as a "classification" head for a gradient reversal task. 

    It gets a feature map in the form of a 3d image
    """
    def __init__(self, in_channels, gr_coeff=1) -> None:
        super(Critic, self).__init__()     

        self.reversal = GradientReversal(gr_coeff)
        self.conv1 = Conv3DBlock(in_channels, 128) # (32,32,4,256) -> (16,16,2,128)
        self.conv2 = Conv3DBlock(128, 256) # (16,16,2,128) -> (8,8,1,256)
        self.conv3 = nn.Sequential(
            nn.Conv3d(256,256,(3,3,1),stride=1,padding=(1,1,0)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d((2,2,1),stride=(2,2,1))
        ) # (8,8,1,256) -> (4,4,1,256)
        self.fc = nn.Sequential(
            nn.Linear(4096,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self, img):      

        img = self.reversal(img)
        out,_ = self.conv1(img)
        out,_ = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.shape[0],-1)
        out = self.fc(out)

        return out
