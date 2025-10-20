
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np


from s2sphere import CellId, LatLng, Cell
import torch.nn as nn
import torch

from random import randint

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose, Resize
from sklearn.model_selection import train_test_split
from time import perf_counter


from geoGuesserDataLoader import GeoGuesserDataset
from map_plot import MapPlotter
from df_prep import load_data
from timer import run_with_timer
from geo_network import GeoNetworkTest, training_loop


CITIES_PATH = "./cities"
STREETVIEW_PATH = "./streetview"
MAPPED_PATH = "./data_mapped"
BATCH_SIZE = 256


transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

    
def main():
    # plot_s2_grid(cells)
    df = load_data(STREETVIEW_PATH, CITIES_PATH, MAPPED_PATH, recompute=False)  

    
    df_img_and_labels = df[['path', 'cell_id']]
    df_img_and_labels, cell_to_idx, idx_to_cell = encode_cells(df_img_and_labels)

    #get a few samples for testing
    array = [randint(1, len(df_img_and_labels)) for i in range(10)]

    df_test = pd.DataFrame(columns=['path', 'cell_id', 'cell_label'])
    for i in range(len(array)):
        df_test.loc[i] = df_img_and_labels.iloc[array[i]]
        df_img_and_labels = df_img_and_labels.drop(i)


    geo_dataloader_train, geo_dataloader_val = small_dataset(df_img_and_labels)
    geo_dataloader_test = test_dataset(df_test)


    # Initialize model, optimizer, and loss function
    first_model = GeoNetworkTest(128)
    optimizer = torch.optim.Adam(first_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Start training
    first_model, train_losses, train_accs, val_losses, val_accs = training_loop(
        first_model, optimizer, loss_fn, geo_dataloader_train, geo_dataloader_val,
        num_epochs=1, print_every=2
    )

    save_path = "./geo_network_test.pth"
    torch.save(first_model.state_dict(), save_path)

    #test
    plotter = MapPlotter(df)
    first_model = torch.load(save_path)
    print
    first_model.eval()

    for i in range(len(geo_dataloader_test)):
        images, labels = next(iter(geo_dataloader_test))
        preds = first_model(images)
        plotter.plot_predictions(images, preds, labels)




def small_dataset(df_img_labels):
    smaller_df = df_img_labels.sample(n=2000, random_state=42)

    df_train, df_val = train_test_split(smaller_df, test_size=0.3)

    dataset_val = GeoGuesserDataset(df_val, transform=transform)
    dataset_train = GeoGuesserDataset(df_train, transform=transform)

    geo_dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)
    geo_dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True,num_workers=4)

    return geo_dataloader_train, geo_dataloader_val

def test_dataset(df_img_labels):

    dataset_test = GeoGuesserDataset(df_img_labels, transform=transform)

    geo_dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4)

    return geo_dataloader_test    


def full_dataset(df_img_labels):
    df_train, df_val = train_test_split(df_img_labels, test_size=0.2)

    dataset_train = GeoGuesserDataset(df_train, transform=transform)
    dataset_val = GeoGuesserDataset(df_val, transform=transform)

    geo_dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    geo_dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return geo_dataloader_train, geo_dataloader_val



def encode_cells(df):
    """Encodes cell_id column into integer class labels."""
    unique_cells = df['cell_id'].unique()
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    idx_to_cell = {idx: cell for cell, idx in cell_to_idx.items()}

    # Add encoded labels as a new column
    df['cell_label'] = df['cell_id'].map(cell_to_idx)

    # tensor = torch.tensor(df['cell_label'].values, dtype=torch.long)
    # one_hot = nn.functional.one_hot(tensor)
    # df['cell_label_one_hot'] = list(one_hot.numpy())

    # df.drop(columns=['cell_label'], inplace=True)

    return df, cell_to_idx, idx_to_cell


if __name__ == "__main__":
    df = load_data(STREETVIEW_PATH, CITIES_PATH, MAPPED_PATH, recompute=False)
    cells = df["cell_id_6"].unique().tolist()   
    main(df, cells=cells, transform=transform)