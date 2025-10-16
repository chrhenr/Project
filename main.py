
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import s2sphere
import torch


from pathlib import Path
from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Normalize, ToTensor, Compose, Resize


from geoGuesserDataLoader import GeoGuesserDataset
from df_prep import load_data


CITIES_PATH = "./cities"
STREETVIEW_PATH = "./streetview"

df = load_data(STREETVIEW_PATH, CITIES_PATH, recompute=False)
# print(f"Totalt antal bilder: {df.index.size}")
cells = df["cell_id"].unique().tolist()

# print(f"unika celler: {len(cells)}")

def encode_cells(df):
    """Encodes cell_id column into integer class labels."""
    unique_cells = df['cell_id'].unique()
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    idx_to_cell = {idx: cell for cell, idx in cell_to_idx.items()}

    # Add encoded labels as a new column
    df['cell_label'] = df['cell_id'].map(cell_to_idx)
    return df, cell_to_idx, idx_to_cell


transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

    
def main(df=df, cells=cells, transform=transform):
    # plot_s2_grid(cells)

    df, cell_to_idx, idx_to_cell = encode_cells(df)
    print(f"Total unique S2 cells: {len(cell_to_idx)}")
    print(df[['cell_id', 'cell_label']].head())

    # unique, y_int = np.unique(df['cell_id'], return_inverse=True)
    # df['cell_id'] = y_int

    dataset = GeoGuesserDataset(df, transform=transform)
    geo_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    print(df.head())

    plot_s2_grid(cells)
    for images, labels in geo_dataloader:
        print(images.shape)
        print(labels.shape)
        break








#Plotting functions
def cell_boundary(cell_id):
    c = Cell(CellId(cell_id))
    verts = []
    for i in range(4):
        v = c.get_vertex(i)
        ll = LatLng.from_point(v)
        verts.append((ll.lat().degrees, ll.lng().degrees))
    return verts + [verts[0]]

def plot_s2_grid(cells):

    plt.figure(figsize=(14, 7))

    # Rita cellernas polygoner (endast celler som faktiskt har data)
    for cid in cells:
        poly = cell_boundary(cid)
        lats = [p[0] for p in poly]
        lons = [p[1] for p in poly]
        plt.plot(lons, lats, linewidth=0.5)  # standardfärger (inga färger sätts)

    # Rita punkter (sampla om datasetet är jättestort)
    N_SAMPLE = min(len(df), 100000)  # höj/sänk vid behov
    sample = df.sample(N_SAMPLE, random_state=0) if len(df) > N_SAMPLE else df
    plt.scatter(sample["lon"], sample["lat"], s=1, alpha=0.7)

    plt.title(f"S2 world grid at level={6} with ALL image points")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.xlim(-180, 180); plt.ylim(-90, 90)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(df=df, cells=cells, transform=transform)