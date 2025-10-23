
from matplotlib.colors import Normalize

from s2sphere import CellId, LatLng
import torch.nn as nn
import torch


from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose, Resize
from sklearn.model_selection import train_test_split


from geoGuesserDataLoader import GeoGuesserDataset
from df_prep import load_data
from geo_network import training_loop, model_ResNet50
from map_plot import MapPlotter


CITIES_PATH = "./cities"
STREETVIEW_PATH = "./streetview"
MAPPED_PATH = "./data_mapped"
EARTH_RADIUS_KM = 6371.0
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20


transform = Compose([
    Resize((168, 168)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class GeoGuesserHelper:
    def __init__(self, recompute, vm):
        self.streetview_path = STREETVIEW_PATH
        self.cities_path = CITIES_PATH
        self.mapped_path = MAPPED_PATH
        self.save_path = "./geo_network_test.pth"
        self.df = load_data(self.streetview_path, self.cities_path, self.mapped_path, recompute=recompute, vm=vm)

        self.df_img_and_labels = None
        self.cell_to_idx = None
        self.idx_to_cell = None

    def prepare_data(self):
        df = self.df.dropna(subset=["path"])  
        self.df_img_and_labels = df[['path', 'cell_id']]
        self.df_img_and_labels, self.cell_to_idx, self.idx_to_cell = encode_cells(self.df_img_and_labels)
    

    def distance_between_cells(self, pred_cell_id, true_cell_id):

        pred_cell = CellId(pred_cell_id)
        true_cell = CellId(true_cell_id)

        pred_center = LatLng.from_point(pred_cell.to_lat_lng().to_point())
        true_center = LatLng.from_point(true_cell.to_lat_lng().to_point())

        distance = EARTH_RADIUS_KM*pred_center.get_distance(true_center).radians
        return distance


    def get_distance(self, prediction, label):

        pred_cell_idx = torch.argmax(prediction, dim=1).item()
        pred_cell_id = self.idx_to_cell[pred_cell_idx]
        true_cell_id = self.idx_to_cell[label]
        distance = self.distance_between_cells(pred_cell_id, true_cell_id)

        return distance


    def test_single_image(self,model, img):
        """Test the model on a single image and return predicted probabilities."""
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = img.convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(image)

        return preds


    def train_model(self, model, train, val, save_path):
        """Train the GeoNetwork model."""
        # Initialize model, optimizer, and loss function
        optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        ], weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        model, train_losses, train_accs, val_losses, val_accs = training_loop(
            model, optimizer, loss_fn, train, val,
            num_epochs=NUM_EPOCHS, print_every=2, save_path=save_path
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    helper = GeoGuesserHelper(recompute=False, vm=True)
    plotter = MapPlotter(helper.df)
    helper.prepare_data()

    plotter.plot_s2_grid()
    # Skapa dataloaders
    geo_dataloader_train, geo_dataloader_val = full_dataset(helper.df_img_and_labels)

    # Train the model
    model = model_ResNet50()
    model.to(device)

    helper.train_model(model, geo_dataloader_train, geo_dataloader_val, helper.save_path)



    # # Test the model
    # model = model_ResNet50()
    # model.load_state_dict(torch.load(helper.save_path, map_location=torch.device('cpu')))
    # model.to(device)

    # geo_dataloader_test = test_dataset(helper.df_img_and_labels)

    # test(model, nn.CrossEntropyLoss(), geo_dataloader_val, device, helper)


def test(model, loss_fn, test_loader, device, helper: GeoGuesserHelper):

    model.eval()
    with torch.no_grad():
        avg_distance = 0.0
        test_acc_cum = 0.0
        test_loss_cum = 0.0

        for x, y in test_loader:
            inputs, labels = x.to(device), y.to(device)
            z = model.forward(inputs)

            batch_loss = loss_fn(z, labels)
            test_loss_cum += batch_loss.item()
            preds = torch.argmax(z, dim=1)
            acc_batch_avg = (preds == labels).float().mean().item()
            test_acc_cum += acc_batch_avg
            # # Compute and print the distance
            # for i in range(len(labels)):
            #     avg_distance += helper.get_distance(preds, labels[i].item())
            # tot_avg_distance = avg_distance/len(labels)

        # print(f"Distance: {tot_avg_distance:.2f} km")
        print(f"Test Loss: {test_loss_cum/len(test_loader):.4f}")
        print(f"Test Accuracy: {test_acc_cum/len(test_loader):.4f}")

    return


def very_small_dataset(df_img_labels):
    """Create a small dataset for quick testing."""
    smaller_df = df_img_labels.sample(n=100, random_state=42)

    df_train, df_val = train_test_split(smaller_df, test_size=0.2)

    dataset_val = GeoGuesserDataset(df_val, transform=transform)
    dataset_train = GeoGuesserDataset(df_train, transform=transform)

    geo_dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)
    geo_dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True,num_workers=4)

    return geo_dataloader_train, geo_dataloader_val


def small_dataset(df_img_labels):
    """Create a small dataset for quick testing."""
    smaller_df = df_img_labels.sample(n=2000, random_state=42)

    df_train, df_val = train_test_split(smaller_df, test_size=0.3)

    dataset_val = GeoGuesserDataset(df_val, transform=transform)
    dataset_train = GeoGuesserDataset(df_train, transform=transform)

    geo_dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)
    geo_dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)

    return geo_dataloader_train, geo_dataloader_val

def test_dataset(df_img_labels):
    """Create a small dataset for quick testing."""
    smaller_df = df_img_labels.sample(n=512, random_state=42)

    dataset_test = GeoGuesserDataset(smaller_df, transform=transform)

    geo_dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4)

    return geo_dataloader_test


def full_dataset(df_img_labels):
    """Create full dataset dataloaders."""
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

    return df, cell_to_idx, idx_to_cell


if __name__ == "__main__":
    main()