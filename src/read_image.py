import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

def read_image(path: Path) -> torch.Tensor:
    img  = Image.open(path)

    img = ImageOps.grayscale(img)
    height = img.height
    width = img.width

    img = torch.from_numpy(np.array(img))
    img = img.reshape(height, width, 1)
    img = img.permute(2, 0, 1)
    
    img = img / 255.0
    return img
