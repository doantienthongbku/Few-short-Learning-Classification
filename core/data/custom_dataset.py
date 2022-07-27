import cv2
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

# Dataset get from: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

class DogCat(Dataset):
    def __init__(self, root, image_pattern='*.jpg', transforms=None):
        self.image_paths = sorted(list(Path(root).glob(image_pattern)), key=lambda x: int(str(x).split('.')[-2]))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # image = cv2.imread(str(image_path))
        # image = cv2.resize(image, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if self.transforms is not None:
        #     image = self.transforms(image=image)['image']
        
        x = Image.open(image_path)
        x = x.resize((28, 28))

        shape = x.size[2], x.size[0], x.size[1]
        x = np.array(x, np.float32, copy=False)
        x = 1.0 - torch.from_numpy(x)
        x = x.transpose(0, 1).contiguous().view(shape)

        if 'cat' in image_path.stem:
            label = 0
        elif 'dog' in image_path.stem:
            label = 1
        else:
            label = -1

        return x, label