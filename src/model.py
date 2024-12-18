from torch import Tensor
import torch.nn as nn

class _Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self._conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self._norm = nn.BatchNorm2d(num_features=out_channels)
        self._relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        result = self._conv1(x)
        result = self._norm(result)
        return self._relu(result)

class GlaucomaClassifier(nn.Module):
    '''
        Input 256 x 256 x 1
        Deep neural network was created based on the network presented in paper: Sułot, Dominika, et al. "Glaucoma classification based on scanning laser ophthalmoscopic images using a deep learning ensemble method." Plos one 16.6 (2021): e0252339
    '''
    def __init__(self) -> None:
        super().__init__()

        self._avg_pool = nn.AvgPool2d(5)

        self._block1_part1 = _Block(1, 8)
        self._block1_part2 = _Block(8, 8)
        self._block1_part3 = _Block(8, 8)

        self._max_pool1 = nn.MaxPool2d(2)

        self._block2_part1 = _Block(8, 16)
        self._block2_part2 = _Block(16, 16)

        self._max_pool1 = nn.MaxPool2d(2)

        self._block3_part1 = _Block(16, 64)
        self._block3_part2 = _Block(64, 64)

        self._max_pool1 = nn.MaxPool2d(2)

        self._flatten = nn.Flatten()
        self._dense1 = nn.Linear(2304, 128) # in:256  out:128
        
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(0.6) 
        self._dense2 = nn.Linear(128, 2) # in:128 out:2
        
        self._softmax = nn.Softmax(dim=1)
    
    def forward(self, x: Tensor) -> Tensor:
        result = self._avg_pool(x)
        result = self._block1_part1(result)
        result = self._block1_part2(result)
        result = self._block1_part3(result)

        result = self._max_pool1(result)

        result = self._block2_part1(result)
        result = self._block2_part2(result)

        result = self._max_pool1(result)

        result = self._block3_part1(result)
        result = self._block3_part2(result)

        result = self._max_pool1(result)

        result = self._flatten(result)
        result = self._dense1(result)
        
        result = self._relu(result)
        result = self._dropout(result)
        result = self._dense2(result)
        
        if self.training:
            return result
        else:
            return self._softmax(result)
