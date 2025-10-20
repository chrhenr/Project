from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from PIL import Image

class GeoGuesserDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        path_to_image = self.dataframe.iloc[idx]['path']

        img = Image.open(path_to_image).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.dataframe.iloc[idx]['cell_label'], dtype=torch.long)

        return img, label

