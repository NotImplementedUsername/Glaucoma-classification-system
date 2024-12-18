import os
import read_image
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor

class EyeFundusDataset(Dataset):
    def __init__(self, location_negative: Path, location_positive: Path) -> None:
        self._location_negative = location_negative
        self._location_positive = location_positive
        self._images = []
        self._labels = []


        for image in os.listdir(self._location_negative):
            f = Path(os.path.join(self._location_negative, image))
            img = read_image.read_image(f)
            self._images.append(img)
            self._labels.append(0)

        for image in os.listdir(self._location_positive):
            f = Path(os.path.join(self._location_positive, image))
            img = read_image.read_image(f)
            self._images.append(img)
            self._labels.append(1)

    def __len__(self) -> int:
        return len(self._images)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._images[idx], self._labels[idx]
