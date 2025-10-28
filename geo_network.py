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
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_features),
        )

    def forward(self, x):
        return self.net(x)
    


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=6, se_ratio=0.25, stride=1):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),  # Swish
            ]

        # Depthwise convolution
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ]

        self.conv = nn.Sequential(*layers)

        # Squeeze & Excitation
        se_hidden = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_hidden, hidden_dim, 1),
            nn.Sigmoid()
        )

        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = x * self.se(x)
        x = self.project(x)
        if self.use_res_connect:
            x = x + identity
        return x

    

class EfficientNetLike(nn.Module):
    def __init__(self, num_classes=1129):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        settings = [
            # in_ch, out_ch, expand, stride, num_blocks
            [32, 16, 1, 1, 1],
            [16, 24, 6, 2, 2],
            [24, 40, 6, 2, 2],
            [40, 80, 6, 2, 3],
            [80, 112, 6, 1, 2],
            [112, 192, 6, 2, 2],
            [192, 320, 6, 1, 1],
        ]

        blocks = []
        for in_ch, out_ch, expand, stride, n in settings:
            for i in range(n):
                blocks.append(MBConv(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    expand_ratio=expand,
                    stride=stride if i == 0 else 1
                ))
            in_ch = out_ch  # update for next stage!

        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    


def unfreeze_last_layer(model):
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

def model_ResNet34(num_classes=1129):
    """Load Resnet34 without pretrained weights and replace the FC layer with a custom head."""
    model = models.resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = Head(num_ftrs, num_classes)
    return model


def model_ResNet50(num_classes=1129):
    # Load base ResNet50 (no pretrained weights yet)
    model = models.resnet50(weights=None)

    # Download pretrained weights from MIT Places
    checkpoint_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

    # Remove the FC weights (they don’t match your head)
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)

    # Load pretrained weights into the backbone
    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        if "fc" not in name:  # Only keep final head trainable
            param.requires_grad = False

    # Replace FC layer with your custom head
    num_ftrs = model.fc.in_features
    model.fc = Head(num_ftrs, num_classes)

    print(sum(p.requires_grad for p in model.parameters()))

    return model



def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every, save_path
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
        torch.save(model.state_dict(), save_path)

        # if epoch == 2:
        #     unfreeze_last_layer(model)
        #     optimizer = torch.optim.Adam(
        #         filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-5
        #     )
        
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


