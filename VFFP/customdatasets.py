import numpy as np
import torch
from skimage.io import imread
from torch.utils import data
from tqdm.notebook import tqdm

class SegmentationData(data.Dataset):
    
    def __init__(self, inputs: list, augment=None):
        self.inputs = inputs
        # self.labels = labels
        self.augment = augment
        self.inputs_dtype = torch.float32
        self.labels_dtype = torch.long
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the pair
        x = self.inputs[index]
        # y = self.labels[index]

        # Preprocessing
        if self.augment is not None:
            x, y = self.augment(x, y)
        
        # Typecasting
        # x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
        #     self.labels_dtype
        # )
        x = torch.from_numpy(x).type(self.inputs_dtype)
        
        # Reshape to fit the dimensions of the updated syntehtic data
        return x
    
    def printitem(self, index:int):
        print(self.inputs[index])
        
        
# Class that returns a 3D cell volume 'x', and the respective labels 'y' for the updated synthetic data
class SegmentationDataSet3D(data.Dataset):
    
    def __init__(self, cells: list, labels: list, augment=None):
        self.cells = cells
        self.labels = labels
        self.augment = augment
        self.cells_dtype = torch.float32
        self.labels_dtype = torch.long
    
    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index: int):
        # Select the pair
        x = self.cells[index]
        y = self.labels[index]

        # Preprocessing
        if self.augment is not None:
            x, y = self.augment(x, y)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.cells_dtype), torch.from_numpy(y).type(
            self.labels_dtype
        )
        
        # Reshape to fit the dimensions of the updated syntehtic data
        return x.reshape(1, 80, 144, 176), y
    
    def printitem(self, index:int):
        print(self.cells[index])

# Class that returns a 3D cell volume 'x', and the respective labels 'y' for the original synthetic data
class CustomSegmentationDataSet(data.Dataset):
    
    def __init__(self, cells: list, labels: list, augment=None):
        self.cells = cells
        self.labels = labels
        self.augment = augment
        self.cells_dtype = torch.float32
        self.labels_dtype = torch.long
    
    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index: int):
        # Select the pair
        x = self.cells[index]
        y = self.labels[index]

        # Preprocessing
        if self.augment is not None:
            x, y = self.augment(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.cells_dtype), torch.from_numpy(y).type(
            self.labels_dtype
        )

        return x.permute(2, 0, 1), y
    
    def printitem(self, index:int):
        print(self.cells[index])

# Class that returns a 3D cell volume 'x', and the respective labels 'y' for the original synthetic data
class Custom3DSegmentationDataSet(data.Dataset):
    
    def __init__(self, cells: list, labels: list, augment=None):
        self.cells = cells
        self.labels = labels
        self.augment = augment
        self.cells_dtype = torch.float32
        self.labels_dtype = torch.long
    
    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index: int):
        # Select the pair
        x = self.cells[index]
        y = self.labels[index]

        # Preprocessing
        if self.augment is not None:
            x, y = self.augment(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.cells_dtype), torch.from_numpy(y).type(
            self.labels_dtype
        )
        
        # Reshape to fit the dimensions of the original syntehtic data
        x = x.reshape(1, 128, 128, 128)

        return x, y
    
    def printitem(self, index:int):
        print(self.cells[index])