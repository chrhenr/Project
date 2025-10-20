from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
import pandas as pd
import torch


class MapPlotter:
    def __init__(self, df):
        self.df = df
        self.cells = df["cell_id"].unique().tolist()

    #Plotting functions
    def cell_boundary(self, cell_id):
        c = Cell(CellId(cell_id))
        verts = []
        for i in range(4):
            v = c.get_vertex(i)
            ll = LatLng.from_point(v)
            verts.append((ll.lat().degrees, ll.lng().degrees))
        return verts + [verts[0]]

    def plot_s2_grid(self):

        plt.figure(figsize=(14, 7))

        # Rita cellernas polygoner (endast celler som faktiskt har data)
        for cid in self.cells:
            poly = self.cell_boundary(cid)
            lats = [p[0] for p in poly]
            lons = [p[1] for p in poly]
            plt.plot(lons, lats, linewidth=0.5)  # standardfärger (inga färger sätts)
            plt.fill_between(lons, lats, alpha=0.1, color='red')  # fyllning för att göra cellerna tydligare


        # Rita punkter (sampla om datasetet är jättestort)
        N_SAMPLE = min(len(self.df), 100000)  # höj/sänk vid behov
        sample = self.df.sample(N_SAMPLE, random_state=0) if len(self.df) > N_SAMPLE else self.df
        plt.scatter(sample["lon"], sample["lat"], s=1, alpha=0.7)

        plt.title(f"S2 world grid at level={6} with ALL image points")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.xlim(-180, 180); plt.ylim(-90, 90)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()

        return "Plotting complete."

    def plot_predictions(self, images, preds, labels):
        probabilities = torch.nn.functional.softmax(preds, dim=1)
        print(probabilities)

