from monai.transforms import MapTransform
import torch


'''
A compilation of monai MapTransforms, used for data processing.
'''

class NormalizeImageToRange(MapTransform):
    """
    Normalize the given input images to a given target range

    """
    def __init__(self, keys, target_range):
      self.keys = keys
      self.target_range = target_range

    def __call__(self,data):
      image = data[self.keys[0]]

      # Map values from the range [a, b] to [c, d]
      a, b = image.min(), image.max()
      c, d = self.target_range[0], self.target_range[1]  # Replace with your desired range

      data[self.keys[0]] = (image - a) * ((d - c) / (b - a)) + c

      return data

class FilterClasses(MapTransform):
    """
    Filter classes from the given label images so that all class not included in the one-hot encoded "classes_to_include" array
    weill be turned to 0 (assumed to mean background).
    """
    def __init__(self, keys, classes_to_include):
      self.keys = keys
      self.classes_to_include = classes_to_include

    def __call__(self, data):
      labels = data[self.keys[1]]

      new_labels = torch.zeros_like(labels)

      classes_indexes = torch.nonzero(self.classes_to_include)[1]

      for index,class_num in enumerate(classes_indexes):
        new_labels[labels==class_num] = index

      data[self.keys[1]] = new_labels

      return data


class CropClass(MapTransform):
    """
    Crops the given label images on the z axis such that a given amount of slices will be a margin around an area
    containing the selected "class_number"
    """

    def __init__(self, keys, class_number, amount_of_slices):
        super().__init__(keys)
        self.keys = keys
        self.class_number = class_number
        self.amount_of_slices = amount_of_slices

    def __call__(self, data):

        labels = data[self.keys[1]]

        class_bin_array = (labels == self.class_number)

        availabel_indexes = torch.any(torch.any(class_bin_array, dim=1),dim=1)[0]
        
        if availabel_indexes.any() == False:
          min = 0
          max = availabel_indexes.shape[0]
        else:
          min = torch.nonzero(availabel_indexes)[0].item()
          max = torch.nonzero(availabel_indexes)[-1].item()

        avg = round((max+min)/2)
        min = avg - int(self.amount_of_slices/2)
        max = avg + int(self.amount_of_slices/2)

        if (min<0):
          min -= min
          max -= min

        for key in self.keys:
          data[key] = data[key][:,:,:,min:max]

        return data