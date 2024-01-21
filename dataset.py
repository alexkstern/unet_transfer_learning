import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset, CacheDataset
import os
import random
from monai.networks.nets import UNet, BasicUNet
import pickle
import numpy as np
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    MapTransform,
)

amos22_classes = [
    "background",
    "spleen",
    "right kidney",
    "left kidney",
    "gall bladder",
    "esophagus",
    "liver",
    "stomach",
    "arota",
    "postcava",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
    "duodenum",
    "bladder",
    "prostate/uterus",
]


baseDir = "Data"
baseDirPelvic = os.path.join(baseDir, "PelvicMRData/data")
baseDirAmos22 = os.path.join(baseDir, "amos22/amos22")


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


def create_img_label_list_PelvicDataset(baseDir):
    return [
        {
            "image": os.path.join(baseDir, f),
            "label": os.path.join(baseDir, f.replace("img", "mask")),
        }
        for f in os.listdir(baseDir)
        if f.endswith("img.nii")
    ]



def create_pelvic_dataset(data_path='./data'):

    # check if .pkl cached dataset already exists
    if not 'pelvic.pkl' in os.listdir('./data'):

        # download the dataset into data folder
        os.system('wget -O "/data/raw/pelvic.zip" "https://zenodo.org/records/7013610/files/data.zip?download=1"')

        # unzip
        os.system()

        pelvic_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.375, 1.375, 5),
                    mode=("bilinear", "nearest"),
                ),
                ResizeWithPadOrCropd(
                    keys=["image", "label"],
                    spatial_size=(128, 128, 16),
                    method="symmetric",
                    mode="constant",
                ),
                NormalizeImageToRange(["image", "label"], [-1, 1]),
            ]
        )

        all_data_list = create_img_label_list_PelvicDataset(baseDirPelvic)
        random.shuffle(all_data_list)

        size = len(all_data_list)
        train_data_list = all_data_list[: int(0.8 * size)]
        val_data_list = all_data_list[int(0.8 * size) :]







def create_amos_dataset(data_path='./data'):

    assert 'amos22' in os.listdir(data_path), 'folder amos22 not found'


    classes_to_include = torch.tensor(
        [[(name == "bladder" or name == "background") for name in amos22_classes]]
    )
    amos22_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropClass(["image", "label"], 14, 32),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.375, 1.375, 5),
                mode=("bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(128, 128, 16),
                method="symmetric",
                mode="constant",
            ),
            FilterClasses(["image", "label"], classes_to_include),
            NormalizeImageToRange(["image", "label"], [-1, 1]),
        ]
    )
