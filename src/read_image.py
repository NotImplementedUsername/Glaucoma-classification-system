import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

def read_image(path: Path) -> torch.Tensor:
    '''
        The function is used to read image and create tensor from its data
    '''
    img  = Image.open(path) # Opening image

    img = ImageOps.grayscale(img)   # Converting image from RGB to gray scale
    height = img.height
    width = img.width

    img = torch.from_numpy(np.array(img))   # Reading image to tensor through numpy array
    img = img.reshape(height, width, 1)     # Adding dimension for channel
    img = img.permute(2, 0, 1)              # Adjusting order of dimensions to expected by PyTorch
    
    img = img / 255.0   # Scaling image to range from 0 to 1
    return img
