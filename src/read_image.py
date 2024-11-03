from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

def read_image(path: Path) -> np.array:
    img  = Image.open(path)

    img = ImageOps.grayscale(img)

    img = np.array(img)
    img = img.astype(np.float32) / 255
    return img