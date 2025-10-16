from torch.utils.data import Dataset, DataLoader
from PIL import Image

class GeoGuesserDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        path_to_image = self.dataframe.iloc[idx]['path']

        img = Image.open(path_to_image)
        if self.transform:
            img = self.transform(img)

        label = self.dataframe.iloc[idx]['cell_id']

        print(f"Image path: {path_to_image}, Cell ID: {label}")
        return img, label

def make_dataLoader(df, transform=None, batch_size=32, shuffle=True, num_workers=4):
    dataset = GeoGuesserDataset(df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)