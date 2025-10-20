import torch
import torch.nn as nn
from torchvision import models


class GeoNetworkBaseline(nn.Module):
    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size  # used to compute the FC input size

        # Two conv layers: 10 filters, k=3, s=1, p=0 (exact spec)
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # After two valid 3×3 convs: H' = W' = D - 4 -- NO LONGER VALID WITH POOLING
        #d_after = img_size - 4
        # in_features = 10 * d_after * d_after  # keep ALL spatial positions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)  # batch size = 1
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 3120),
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        x = self.features(input_batch)          # (B, 10, D-4, D-4)
        x = x.reshape(x.size(0), -1)            # (B, 10*(D-4)*(D-4))
        x = self.classifier(x)                  # (B, 3120)
        return x                                # (B, 3120)



class Head(nn.Module):
    """The classification head."""
    def __init__(self, input_features, output_features):
        super().__init__()
        self.in_features = input_features
        self.out_features = output_features

        self.classifier = nn.Sequential(
        nn.Linear(self.in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, self.out_features),
        )

    def forward(self, input_batch):
        x = input_batch.reshape(input_batch.size(0), -1)
        x = self.classifier(x)
        return x.squeeze(-1)


def train_with_ResNet50(img_size: int) -> nn.Module:
    """Create a GeoNetwork model based on ResNet50."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = Head(num_ftrs, 3120)  # Adjust the final layer for 3120 classes
    return model


def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every
):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        model, train_loss, train_acc = train_epoch(
            model, optimizer, loss_fn, train_loader, val_loader, device, print_every
        )
        val_loss, val_acc = validate(model, loss_fn, val_loader, device)
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
            f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
            f"Val. loss: {val_loss:.3f}, "
            f"Val. acc.: {val_acc:.3f}"
        )
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, train_accs, val_losses, val_accs


def train_epoch(
    model, optimizer, loss_fn, train_loader, val_loader, device, print_every
):
    # Train:
    model.train()
    train_loss_batches, train_acc_batches = [], []
    num_batches = len(train_loader)
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)
        optimizer.zero_grad()
        z = model.forward(inputs)
        loss = loss_fn(z, labels)
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

        preds = torch.argmax(z, dim=1)
        acc_batch_avg = (preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)


        if print_every is not None and batch_index % print_every == 0:
            val_loss, val_acc = validate(model, loss_fn, val_loader, device)
            model.train()
            print(
                f"\tBatch {batch_index}/{num_batches}: "
                f"\tTrain loss: {sum(train_loss_batches[-print_every:])/print_every:.3f}, "
                f"\tTrain acc.: {sum(train_acc_batches[-print_every:])/print_every:.3f}, "
                f"\tVal. loss: {val_loss:.3f}, "
                f"\tVal. acc.: {val_acc:.3f}"
            )

    return model, train_loss_batches, train_acc_batches


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            z = model.forward(inputs)

            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()
            preds = torch.argmax(z, dim=1)
            acc_batch_avg = (preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg

    return val_loss_cum / len(val_loader), val_acc_cum / len(val_loader)


