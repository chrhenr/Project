from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import numpy as np
import torch


class MapPlotter:
    def __init__(self, df):
        self.df = df
        self.cells = df["cell_id"].unique().tolist()

    #Plotting functions
    def cell_boundary(self, cell_id):
        """Return the boundary vertices of an S2 cell given its cell ID."""
        c = Cell(CellId(cell_id))
        verts = []
        for i in range(4):
            v = c.get_vertex(i)
            ll = LatLng.from_point(v)
            verts.append((ll.lat().degrees, ll.lng().degrees))
        return verts + [verts[0]]

    def plot_s2_grid(self, probabilities=None, image=None):
        """Plot the S2 grid cells and optionally color them based on probabilities."""
        plt.figure(figsize=(14, 7))

        # Rita punkter (sampla om datasetet är jättestort)
        N_SAMPLE = min(len(self.df), 50000)  # höj/sänk vid behov
        sample = self.df.sample(N_SAMPLE, random_state=0) if len(self.df) > N_SAMPLE else self.df
        plt.scatter(sample["lon"], sample["lat"], s=1, alpha=0.7)

        # Rita cellernas polygoner (endast celler som faktiskt har data)
        for index, cid in enumerate(self.cells):
            poly = self.cell_boundary(cid)

            lats = [p[0] for p in poly]
            lons = [p[1] for p in poly]
            polygon = Polygon(list(zip(lons, lats)), closed=True, linewidth=0.5, edgecolor=(0, 0, 0, 1.0), facecolor=(1, 0, 0, probabilities[index] if probabilities is not None else 0.0))
            plt.gca().add_patch(polygon)
        
        plt.imshow(plt.imread(image))
        plt.show()

        plt.title(f"S2 world grid at level={6} with ALL image points")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.xlim(-180, 180); plt.ylim(-90, 90)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()

        return "Plotting complete."

    def plot_predictions(self, image_index, logits):
        """Plot the S2 grid with predicted probabilities for a given image index."""
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        image = self.df.iloc[image_index]['path']
    
        probabilities = probabilities*(probabilities > 0.01).float()
        predicted = probabilities.detach().numpy()
    
        predicted = predicted[0]
        predicted[np.argmax(predicted)] = 1.0  # Highlight the most probable cell
        self.plot_s2_grid(predicted, image)

