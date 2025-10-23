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

    def plot_s2_grid(self, probabilities=None):
        # Load background image

        # im = plt.imread("mapOption.png")

        fig, ax = plt.subplots(figsize=(14, 7))
        # ax.set_aspect("equal", adjustable="box")

        # Sample points for plotting
        N_SAMPLE = min(len(self.df), 50000)
        sample = self.df.sample(N_SAMPLE, random_state=0) if len(self.df) > N_SAMPLE else self.df
        ax.scatter(sample["lon"], sample["lat"], s=1, alpha=0.7)

        # Draw polygons for each S2 cell
        for index, cid in enumerate(self.cells):
            poly = self.cell_boundary(cid)
            lats = [p[0] for p in poly]
            lons = [p[1] for p in poly]
            polygon = Polygon(
                list(zip(lons, lats)),
                closed=True,
                linewidth=0.5,
                edgecolor=(0, 0, 0, 0.5),
                facecolor=(1, 0, 0, probabilities[index] if probabilities is not None else 0.0)
            )
            ax.add_patch(polygon)

        # Set geographic extent and add map background
        ax.imshow(im, extent=[-180, 180, -90, 90], origin="upper")

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"S2 world grid at level={6} with ALL image points")

        plt.tight_layout()
        plt.show()
        return "Plotting complete."


    def plot_predictions(self, logits):
        """Plot the S2 grid with predicted probabilities for a given image index."""
        probabilities = torch.nn.functional.softmax(logits, dim=1)


        # probabilities = probabilities*(probabilities > 0.01).float()
        predicted = probabilities.detach().numpy()
    
        predicted = predicted[0]
        predicted[np.argmax(predicted)] = 1.0  # Highlight the most probable cell
        self.plot_s2_grid(predicted)


    def plot_embellished_predictions(self, logits):
            """Plot the S2 grid with predicted probabilities for a given image index, with embellishments."""
            probabilities = torch.nn.functional.softmax(logits, dim=1)

            predicted = probabilities.detach().numpy()
        
            predicted = predicted[0]

            predicted = predicted*1000 
            predicted[np.argmax(predicted)] = 1.0  # Highlight the most probable cell


            # Load and display a background map image
            # map_image_path = "world_map.png"  # Path to your world map image
            self.plot_s2_grid(predicted)