from torch.utils.data import DataLoader, Subset
import torch
import torch.optim as optim
import random
from monai.losses import DiceLoss
from config import config
import pickle
import os
from models import UNet3D
from train_and_evaluate_loops import train_and_evaluate,pickle_write

random.seed(42)
baseDirPelvicMR = config.baseDirPelvic
baseDir  = config.baseDir

pelvicMR_classes = [
    'none',
    'bladder',
    'bone',
    'obturator internus',
    'transition zone',
    'central gland',
    'rectum',
    'seminal vesicle',
    'neurovascular bundle'
]

params = {
    'batch_size': 8,
    'lr': 0.0001,
    'optimizer': 'adam',
    'channels': [64, 128],
    'bottleneck_channels': 256,
    'loss_include_backgroud': False,
    'augmentations': True,
    'augmentation_max_shift':32,
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Datasets ~~~~~~~~~~~~~~~~~~~~~~~~

with open(os.path.join(baseDir,'train_dataset_pelvic_128.pkl'), 'rb') as f:
    train_dataset = pickle.load(f)

with open(os.path.join(baseDir,'valid_dataset_pelvic_128.pkl'), 'rb') as f:
    not_train_dataset = pickle.load(f)


not_train_indices = list(range(len(not_train_dataset)))
random.shuffle(not_train_indices)

# split the remaining 20% of the dataset into 10% - validation, 10% - test
valid_indices = not_train_indices[:len(not_train_indices)//2]
test_indices = not_train_indices[len(not_train_indices)//2:]

print(valid_indices)
print(test_indices)

valid_dataset = Subset(not_train_dataset, valid_indices)
test_dataset = Subset(not_train_dataset, test_indices)

print('train_size: ',len(train_dataset))
print('valid_size: ',len(valid_dataset))
print('test_size: ',len(test_dataset))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DataLoaders ~~~~~~~~~~~~~~~~~~~~~~~~
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_classes = len(pelvicMR_classes)

model = UNet3D(1,len(pelvicMR_classes),params['channels'],params['bottleneck_channels']).to(device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Criterion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

criterion = DiceLoss(to_onehot_y=True,softmax=True,include_background=params['loss_include_backgroud'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if params['optimizer'] == 'adam':
  optimizer = optim.Adam(model.parameters(), lr=params['lr'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Augmentations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

augmentations = None

def shift_img_batch(imgs,x,y):
  if x!=0:
    imgs = torch.roll(imgs,shifts=x,dims=2)
    if x>0:
      imgs[:,:,:x,:,:] = 0
    else:
      imgs[:,:,x:,:,:] = 0

  if y!=0:
    imgs = torch.roll(imgs,shifts=y,dims=3)
    if y>0:
      imgs[:,:,:,:y,:] = 0
    else:
      imgs[:,:,:,y:,:] = 0

  return imgs

def augmentations_f(imgs,masks):

  max_shift = params['augmentation_max_shift']

  shiftx = torch.randint(-max_shift,max_shift+1,[1]).item()
  shifty = torch.randint(-max_shift,max_shift+1,[1]).item()

  imgs = shift_img_batch(imgs,shiftx,shifty)
  masks = shift_img_batch(masks,shiftx,shifty)

  return imgs,masks


if params['augmentations']:
  augmentations = augmentations_f

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(params)
best_model, train_loss_list, val_loss_list, val_dice_list = train_and_evaluate(model,num_classes, train_loader, valid_loader, valid_dataset, criterion, optimizer, device, 100, 5, augmentations, params)

# torch.save(best_model, os.path.join(baseDir,'pelvic_best_model_unet3d.pth'))
torch.save(best_model.state_dict(), os.path.join(baseDir,'pelvic_best_model_unet3d_state_dict_{}.pth'.format(params['channels'])))

# torch.save(best_model, os.path.join(baseDir,'pelvic_best_model.pth'))
pickle_write((train_loss_list,val_loss_list,val_dice_list,params),os.path.join(baseDir,'pelvic_results_unet3d_{}.pkl'.format(params['channels'])))

