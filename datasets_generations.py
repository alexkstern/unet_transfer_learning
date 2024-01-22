"""
As a result of hardware limitations we had to mainly use google colab, which is not really efficient at reading files.
To compensate for that we used monai's CachedDatasets so that we only had to compile the ds once and then we could reuse them.
(Loading the single chached dataset is much faster on google colab than loading all images one after the other)

here we create those datasets.
"""

import os
import pandas as pd
from transforms import NormalizeImageToRange
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
)
from monai.data import CacheDataset
from transforms import CropClass,FilterClasses
from config import config
import pickle
import torch
import random

baseDirAmos22 = config.baseDirAmos22
baseDir = config.baseDir
baseDirPelvic = config.baseDirPelvic

# when generating the datasets we used: random.seed(42)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1st Dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_img_label_list_PelvicDataset(baseDir):
  return [{'image':os.path.join(baseDir,f),'label':os.path.join(baseDir,f.replace('img','mask'))} for f in os.listdir(baseDir) if f.endswith('img.nii')]

pelvic_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.375, 1.375, 5), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image","label"],spatial_size=(128,128,16),method="symmetric",mode="constant"),
        NormalizeImageToRange(["image","label"],[-1,1])
    ])

def save_cached_datasets_for_pelvicMR():
    all_data_list = create_img_label_list_PelvicDataset(baseDirPelvic)
    random.shuffle(all_data_list)

    size = len(all_data_list)
    train_data_list = all_data_list[:int(0.8*size)]
    val_data_list =  all_data_list[int(0.8*size):]

    num_workers = os.cpu_count()
    print("workers: {}".format(num_workers))

    train_dataset = CacheDataset(data=train_data_list, transform=pelvic_transforms, num_workers=1)
    valid_dataset = CacheDataset(data=val_data_list, transform=pelvic_transforms, num_workers=1)

    save_path = os.path.join(baseDir,'train_dataset_pelvic_128.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(train_dataset, f)

    save_path = os.path.join(baseDir,'valid_dataset_pelvic_128.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(valid_dataset, f)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        "prostate/uterus"
]

CT_scanners_amos22 = ['Aquilion ONE',
 'Optima CT660',
 'SOMATOM Force',
 'Achieva',
 'Brilliance16',
 'Optima CT540'
 ]

MRI_scanners_amos22 = [
    'Ingenia',
    'Prisma',
    'SIGNA HDe'
 ]

classes_to_include = torch.tensor([[(name == "bladder" or name=="background") for name in amos22_classes]])
amos22_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropClass(["image","label"],14,32),
        Spacingd(keys=["image", "label"], pixdim=(1.375, 1.375, 5), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image","label"],spatial_size=(128,128,16),method="symmetric",mode="constant"),
        FilterClasses(["image","label"],classes_to_include),
        NormalizeImageToRange(["image","label"],[-1,1])
    ])

def create_img_label_list_amos22_df(df):
  images_labels = df[['image path','labels path']]
  return [{'image':os.path.join(baseDirAmos22,f[1][0][1:]),'label':os.path.join(baseDirAmos22,f[1][1][1:])} for f in images_labels.iterrows()]

def create_image_label_list_amos22(baseDir, label_path_filter, model_name_filter):
  amos22_df = pd.read_csv(os.path.join(baseDir,'labeled_data_meta_0000_0599_with_paths.csv'))
  amos22_df = amos22_df[[label_path_filter(label) for label in amos22_df['labels path']]]
  amos22_df = amos22_df[[model_name_filter(model) for model in amos22_df['Manufacturer\'s Model Name']]]
  amos22_df.reset_index(inplace=True)
  return create_img_label_list_amos22_df(amos22_df)

def save_cached_datasets_for_amos22():
  train_data_list = create_image_label_list_amos22(baseDirAmos22, lambda path: path.find('Tr')!=-1, lambda model: model in CT_scanners_amos22)
  val_data_list = create_image_label_list_amos22(baseDirAmos22, lambda path: path.find('Va')!=-1, lambda model: model in CT_scanners_amos22)

  train_dataset = CacheDataset(data=train_data_list, transform=amos22_transforms)
  valid_dataset = CacheDataset(data=val_data_list, transform=amos22_transforms)

  save_path = os.path.join(baseDir,'train_dataset_amos22_128.pkl')
  with open(save_path, 'wb') as f:
      pickle.dump(train_dataset, f)

  save_path = os.path.join(baseDir,'valid_dataset_amos22_128.pkl')
  with open(save_path, 'wb') as f:
      pickle.dump(valid_dataset, f)