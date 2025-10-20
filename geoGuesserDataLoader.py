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
        # One of the images gave an error when loading
        try:
            img = Image.open(path_to_image).convert("RGB")
        except Exception as e:
            print(f"\nException occurred while opening image at index {idx}:")
            print("path_to_image:", path_to_image)
            print("Type of path_to_image:", type(path_to_image))
            print("Exception message:", e)
            raise e  # re-raise the error so training fails, or you can choose to return a dummy image here
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.dataframe.iloc[idx]['cell_label'], dtype=torch.long)

        return img, label

