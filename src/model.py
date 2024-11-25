from torch import Tensor
import torch.nn as nn
import torch

class Block1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        result = self.conv1(x)
        result = self.norm(result)
        return self.relu(result)

class GlaucomaClassifier(nn.Module):
    '''
        Input 256 x 256 x 1
        Deep neural network was created based on the network presented in paper: Sułot, Dominika, et al. "Glaucoma classification based on scanning laser ophthalmoscopic images using a deep learning ensemble method." Plos one 16.6 (2021): e0252339
    '''
    def __init__(self) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(5)

        self.block1_part1 = Block1(1, 8)
        self.block1_part2 = Block1(8, 8)
        self.block1_part3 = Block1(8, 8)

        self.max_pool1 = nn.MaxPool2d(2)

        self.block2_part1 = Block1(8, 16)
        self.block2_part2 = Block1(16, 16)

        self.max_pool1 = nn.MaxPool2d(2)

        self.block3_part1 = Block1(16, 64)
        self.block3_part2 = Block1(64, 64)

        self.max_pool1 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2304, 128) # in:256  out:128
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6) 
        self.dense2 = nn.Linear(128, 2) # in:128 out:2
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: Tensor) -> Tensor:
        result = self.avg_pool(x)
        result = self.block1_part1(result)
        result = self.block1_part2(result)
        result = self.block1_part3(result)

        result = self.max_pool1(result)

        result = self.block2_part1(result)
        result = self.block2_part2(result)

        result = self.max_pool1(result)

        result = self.block3_part1(result)
        result = self.block3_part2(result)

        result = self.max_pool1(result)

        result = self.flatten(result)
        result = self.dense1(result)
        
        result = self.relu(result)
        result = self.dropout(result)
        result = self.dense2(result)
        
        if self.training:
            return result
        else:
            return self.softmax(result)
