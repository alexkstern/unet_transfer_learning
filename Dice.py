import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self, num_classes):
        super(Dice, self).__init__()
        self.num_classes = num_classes

    def forward(self, predicted, target):

      predicted = torch.nn.functional.one_hot(predicted.long(),self.num_classes).permute((0,4,1,2,3))
      one_hot_tar = torch.nn.functional.one_hot(target.long(), self.num_classes).permute((0,4,1,2,3))

      pred_flat = predicted.view(predicted.size(0), self.num_classes,-1)
      tar_flat = one_hot_tar.view(one_hot_tar.size(0), self.num_classes,-1)

      dice_numinator = 2*(pred_flat*tar_flat).sum(2)
      dice_denominator = (pred_flat.sum(2) + tar_flat.sum(2))

      dice_denominator_zeros = (dice_denominator == 0)

      # in cases where both pred and tar have no area, we set the dice to 1
      dice_numinator[dice_denominator_zeros] = 1
      dice_denominator[dice_denominator_zeros] = 1

      dice = (dice_numinator/dice_denominator).mean(0)

      return dice