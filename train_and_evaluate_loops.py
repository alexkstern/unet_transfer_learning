import torch
import matplotlib.pyplot as plt
import datetime
from Dice import Dice
import json
import os
from config import config
import copy
import csv
import pickle

baseDir = config.baseDir

def pickle_read(path):
  with open(path, 'rb') as f:
      data = pickle.load(f)
  return data

def pickle_write(data,path):
  with open(path, 'wb') as f:
      pickle.dump(data, f)

def append_to_csv(csv_path, numbers):
    """
    Helper function for appending lines to csv files (mainly used to report results while training)
    """
    try:
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(numbers)
    except Exception as e:
        print(f"Error: {e}")

def write_dic(dic,filepath):
  """
  Helper function to save a 'json' dictionary (mainly used to document params of training run)
  """  
  json_data = json.dumps(dic, indent=4)  # indent parameter for pretty formatting
  
  with open(filepath, 'w') as file:
      file.write(json_data)


def print_batch(imgs,labels,pred):
  """
  Helper function to print a batch of segmentations, to assist in visually inspecting the model performance
  """

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


def train_one_epoch(model, dataloader, criterion, optimizer, device, augmentations=None):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0    

    

    for batch_data in dataloader:

        imgs, masks = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )

        if augmentations is not None:
          imgs, masks = augmentations(imgs,masks)

        optimizer.zero_grad()
        outputs, _ = model(imgs)

        predictions = torch.argmax(outputs,dim=1)
        accuracy = (predictions == masks.squeeze(1)).sum()/torch.numel(masks)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

    print_batch(imgs,masks,outputs)

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)
    return average_loss, average_accuracy


def evaluate(model, num_classes, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    dice_metric = Dice(num_classes)

    with torch.no_grad(): 
        
        for batch_data in dataloader:

            imgs, masks = (
              batch_data["image"].to(device),
              batch_data["label"].to(device),
            )

            outputs,_ = model(imgs)

            predictions = torch.argmax(outputs,dim=1)
            accuracy = (predictions == masks.squeeze(1)).sum()/torch.numel(masks)
            dice = dice_metric(predictions,masks.squeeze(1))

            loss = criterion(outputs, masks.long())

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_dice += dice


    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)
    average_dice = total_dice/len(dataloader)
    return average_loss, average_accuracy, average_dice

def get_current_time_string():
    now = datetime.datetime.now()
    time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    return time_string


def train_and_evaluate(model,num_classes, train_dataloader, val_dataloader, val_dataset, criterion, optimizer, device, num_epochs=10, number_of_non_improvement = 5, augmentations = None, params = None):

    best_model = {}
    min_loss = 9999999

    timestring = get_current_time_string()
    write_dic(params,os.path.join(baseDir,'params_layers_{}_{}_{}.json'.format(params['base_name'],params['channels'],timestring)))
    csv_path = os.path.join(baseDir,'results_csv_{}_layers_{}_{}.csv'.format(params['base_name'],params['channels'],timestring))

    failed_to_improve_count = 0

    train_loss_list = []
    val_loss_list = []
    val_dice_list = []

    for epoch in range(num_epochs):

        torch.cuda.empty_cache()

        # Training
        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, criterion, optimizer, device, augmentations)
        print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        # Evaluation
        val_loss, val_accuracy, val_dice = evaluate(model,num_classes, val_dataloader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        print(val_dice)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_dice_list.append(val_dice)

        append_to_csv(csv_path,[train_loss,val_loss,val_dice.tolist()])

        if val_loss < min_loss:
          print('new best model')
          min_loss = val_loss
          best_model = copy.deepcopy(model)
          failed_to_improve_count = 0
        else:
          failed_to_improve_count+=1

        if failed_to_improve_count == number_of_non_improvement:
          break

    return best_model, train_loss_list, val_loss_list, val_dice_list