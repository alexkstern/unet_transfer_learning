

from torch.utils.data import DataLoader,  Subset
import torch
import torch.optim as optim
import random
from monai.losses import DiceLoss
from models import UNet3D, OutputChannel3DConverter
import pickle
import os
from config import config
from train_and_evaluate_loops import train_and_evaluate, pickle_write

baseDir = config.baseDir

random.seed(42)


def get_model_for_amos(device, params ,path=None):

    unet = UNet3D(1,9,params['channels'],params['bottleneck_channels']).to(device)

    # Print initial weights of the model
    # print_initial_weights(model)

    if path is not None:
        pretrained_dict = torch.load(path)

        # Update the model's weights with the pretrained weights
        unet.load_state_dict(pretrained_dict)

    model = OutputChannel3DConverter(unet, 9, 2).to(device)

    return model

def print_initial_weights(model):
    print("Model parameters (part of):")
    for name, param in model.named_parameters():
        if 'conv' in name:
            print(f"{name}: {param[0][0][0][0][0]}")
            break


params = {
    'batch_size': 4,
    'lr': 0.00001,
    'optimizer': 'adam',
    'channels': [64, 128, 256],
    'bottleneck_channels': 512,
    'loss_include_backgroud': False,
    'augmentations': True,
    'augmentation_max_shift':32,
    'train_size':25,
    'base_name':'fine_tune'
}
params['base_name'] = '{}_size_{}_lr_{}'.format(params['base_name'],params['train_size'],params['lr'])

num_classes = 2

with open(os.path.join(baseDir,'train_dataset_amos22_128.pkl'), 'rb') as f:
    train_dataset = pickle.load(f)

with open(os.path.join(baseDir,'valid_dataset_amos22_128.pkl'), 'rb') as f:
    valid_dataset = pickle.load(f)
  

print('loaded_train_size: ',len(train_dataset))
print('loaded_valid_size: ',len(valid_dataset))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train Subset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# train_indices = random.sample(range(len(train_dataset)), config.train_size)
train_indices = list(range(len(train_dataset)))
random.shuffle(train_indices)
train_indices = train_indices[:params['train_size']]

print('random indices: ',train_indices)

train_subset = Subset(train_dataset, train_indices)

print('train_subset size:', len(train_subset))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DataLoaders ~~~~~~~~~~~~~~~~~~~~~~~~

train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the model with pretrained weights
model_transfer = get_model_for_amos(device, params, os.path.join(baseDir,'pelvic_best_model_unet3d_state_dict_{}.pth'.format(params['channels'])))
model_transfer.baseModel.freeze_decoder_layers()
model_scratch = get_model_for_amos(device, params)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Criterion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

criterion = DiceLoss(to_onehot_y=True,softmax=True,include_background=params['loss_include_backgroud'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
optimizer_transfer = optim.Adam([ param for param in model_transfer.parameters() if param.requires_grad == True], lr=params['lr'])
optimizer_scratch = optim.Adam([ param for param in model_scratch.parameters() if param.requires_grad == True], lr=params['lr'])

orig_base_name = params['base_name']



params['base_name'] = '{}_{}'.format(orig_base_name,'scratch')
best_model_scratch, train_loss_list_scratch, val_loss_list_scratch, val_dice_list_scratch =\
 train_and_evaluate(model_scratch,num_classes, train_loader, valid_loader, valid_dataset, criterion, optimizer_scratch, device, 100,10,None,params)



params['base_name'] = '{}_{}'.format(orig_base_name,'transfer')
best_model_transfer, train_loss_list_transfer, val_loss_list_transfer, val_dice_list_transfer =\
 train_and_evaluate(model_transfer,num_classes, train_loader, valid_loader, valid_dataset, criterion, optimizer_transfer, device, 100,10,None,params)


# torch.save(best_model, os.path.join(baseDir,'amos22_transferl_best_model.pth'))
pickle_write((train_loss_list_scratch,val_loss_list_scratch,val_dice_list_scratch),os.path.join(baseDir,f'amos22_results_{params["base_name"]}_scratch.pkl'))
pickle_write((train_loss_list_transfer,val_loss_list_transfer,val_dice_list_transfer),os.path.join(baseDir,f'amos22_results_{params["base_name"]}_transfer.pkl'))
