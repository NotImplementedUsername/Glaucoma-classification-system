import os
import read_image
from pathlib import Path
from torch.utils.data.dataset import Dataset

class Eye_fundus_dataset(Dataset):
    def __init__(self, location_negative: Path, location_positive: Path) -> None:
        self.location_negative = location_negative
        self.location_positive = location_positive
        self.images = []
        self.labels = []


        for image in os.listdir(self.location_negative):
            f = Path(os.path.join(self.location_negative, image))
            img = read_image.read_image(f)
            self.images.append(img)
            self.labels.append(0)

        for image in os.listdir(self.location_positive):
            f = Path(os.path.join(self.location_positive, image))
            img = read_image.read_image(f)
            self.images.append(img)
            self.labels.append(1)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> None:
        return self.images[idx], self.labels[idx]