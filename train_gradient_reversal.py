import torch.optim as optim
from monai.losses import DiceLoss, Dataset
import torch
import pickle
from config import config
import matplotlib.pyplot as plt
import os
from monai.data import DataLoader
from models import UNet3D, Critic
import torch.nn as nn
import math

baseDir = config.baseDir

batch_size = 4
num_classes = 9
num_epochs = 12000
critic_clipping = 0.03
beta = 1
gradient_inversal_coeff = 0.1

class CombinedDataset(Dataset):
    """
    Helper class to make one dataset that simultaneously query all of the given datasets.
    It will assume the length of the biggest ds and will cyclically load from other ones.
    """
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        max = 0
        for ds in self.datasets:
          if len(ds)>max:
            max = len(ds)

        return max

    def __getitem__(self, index):

        indexes = [index%len(ds) for ds in self.datasets]

        items = [ds.__getitem__(i) for i,ds in zip(indexes,self.datasets)]

        return list(items)


def print_batch(imgs,labels,pred):

  batch_size = torch.min(torch.tensor([imgs.shape[0],5])).item()
  pred = torch.argmax(pred,dim=1)

  plt.figure()
  for i in range(batch_size):
    plt.subplot(3,batch_size,i+1)
    plt.imshow(imgs[i][0].detach().cpu().numpy()[:,:,8],cmap='gray')
    plt.subplot(3,batch_size,i+1+batch_size)
    plt.imshow(labels[i][0].detach().cpu().numpy()[:,:,8],vmin=0,vmax=8)
    plt.subplot(3,batch_size,i+1+2*batch_size)
    plt.imshow(pred[i].detach().cpu().numpy()[:,:,8],vmin=0,vmax=8)
  plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Datasets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

with open(os.path.join(baseDir,'train_dataset_pelvic_128.pkl'), 'rb') as f:
      ds_pelvic_train = pickle.load(f)

with open(os.path.join(baseDir,'valid_dataset_pelvic_128.pkl'), 'rb') as f:
      de_pelvic_valid = pickle.load(f)

with open(os.path.join(baseDir,'train_dataset_amos22_128.pkl'), 'rb') as f:
      ds_amos_train = pickle.load(f)

with open(os.path.join(baseDir,'valid_dataset_amos22_128.pkl'), 'rb') as f:
      de_amos_valid = pickle.load(f)

# ~~~~~~~~~~~~~~~~~~~~~~ Data Loaders ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dl_pelvic_train = DataLoader(ds_pelvic_train, batch_size=batch_size, shuffle=True, num_workers=1)
dl_amos_train = DataLoader(ds_amos_train, batch_size=batch_size, shuffle=True, num_workers=1)

print('pelvic train size: {}'.format(len(ds_pelvic_train)))
print('amos train size: {}'.format(len(ds_amos_train)))

combined_ds_train = CombinedDataset([ds_pelvic_train,ds_amos_train])
combined_loader_train = DataLoader(combined_ds_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last = True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


model = UNet3D(1,num_classes)
model.to(device)

critic = Critic(256,gradient_inversal_coeff).to(device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

optimizer = optim.Adam(list(critic.parameters())+list(model.parameters()),lr=0.003)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dice_loss = DiceLoss(to_onehot_y=True,softmax=True,include_background=False)
bce = nn.BCELoss()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
epoch_i = 0
for epoch_i in range(num_epochs):  
  for items in combined_loader_train:

    torch.cuda.empty_cache()

    model.eval()
    critic.train()

    outputs_pelvic = None
    features_pelvic = None

    outputs_amos = None
    features_amos = None

  # ~~~~~~~~~~~~~~~~~~~~~~ Pred and Feeatures: pelvic ~~~~~~~~~~~~~~~~~~~~~~~~~

    pelvic_imgs, pelvic_masks = (
              items["image"][0:batch_size*2:2].to(device),
              items["label"][0:batch_size*2:2].to(device),
    )

  # ~~~~~~~~~~~~~~~~~~~~~~ Pred and Features: amos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    amos_imgs, amos_masks = (
              items["image"][1:batch_size*2:2].to(device),
              items["label"][1:batch_size*2:2].to(device),
    )

  # ~~~~~~~~~~~~~~~~~~~~~~~~~ Train Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    optimizer.zero_grad()

    # pass both classes thorough Unet    
    outputs_pelvic, features_pelvic = model(pelvic_imgs)
    outputs_amos, features_amos = model(amos_imgs)
    
    # concat results
    concat_both = torch.concat((features_pelvic, features_amos),dim=0)

    # classify with classifier
    critic_scores = critic(concat_both)

    # prepare labels to asses classifier performance
    ones = torch.ones((batch_size,1)).to(device)
    zeros = torch.zeros((batch_size,1)).to(device)
    labels = torch.concat((ones,zeros),dim=0)

    # asses classifier performance with BCE
    critic_loss = bce(critic_scores,labels)

    # asses Unet classification of pelvic images
    model_dice_loss = dice_loss(outputs_pelvic,pelvic_masks.long())    

    # compute loss (remember gradient from critic is reversed when propagating to model)
    loss = model_dice_loss + beta*critic_loss
    loss.backward()

    optimizer.step()    

  # ~~~~~~~~~~~~~~~~~~~~~~~~ Print ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  print_batch(amos_imgs,amos_masks,outputs_amos)
  print_batch(pelvic_imgs,pelvic_masks,outputs_pelvic)
  print(critic_scores)
  print('epoch {}, loss: {:.4f}'.format(epoch_i,loss.item()))
  print('model dice: {:.4f}'.format(model_dice_loss))
  print('critic loss: {:.4f}'.format(critic_loss.item()))

