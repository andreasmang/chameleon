import os, random, math, numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")   # apple silicon GPU (macOS 12.3+)
    else:
        return torch.device("cpu")   # fallback to CPU


# define seed; make things reproducible
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# load mnist data
def get_loaders(data_dir="./data", batch_size=128, val_frac=0.2, seed=42):
    tfm = transforms.ToTensor()  # yields [0,1], shape [1,28,28]
    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    val_size = int(len(full_train) * val_frac)
    train_size = len(full_train) - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=g)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    )

# define mlp (neural network architecture)
class MLP(nn.Module):
    def __init__(self, dropout_p=0.2, use_bn=False):
        super().__init__()
        layers = []
        def block(in_f, out_f):
            blk = [nn.Linear(in_f, out_f)]
            if use_bn: blk += [nn.BatchNorm1d(out_f)]
            blk += [nn.ReLU(), nn.Dropout(p=dropout_p)]
            return blk
        layers += block(28*28, 256)
        layers += block(256, 128)
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        if x.ndim == 4:  # [B,1,28,28]
            x = x.view(x.size(0), -1)  # flatten
        x = self.features(x)
        return self.head(x)

# evaluate accuracy
@torch.no_grad()
def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

# execute epoch
def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train: optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        total_n += bs
    return total_loss / total_n, total_acc / total_n


# training
def train(
    epochs=10,
    batch_size=128,
    lr=1e-3,
    optimizer_name="adam",
    dropout_p=0.2,
    use_bn=False,
    seed=42,
    save_path="best.pt"):

    set_seed(seed)
    device = get_device()
    print("device is", device)

    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size, seed=seed)

    # setup architecutres
    model = MLP(dropout_p=dropout_p, use_bn=use_bn).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # run epochs
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(val_loss)
        hist["train_acc"].append(tr_acc);   hist["val_acc"].append(val_acc)
        print(f"epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "hparams": dict(epochs=epochs,batch_size=batch_size,lr=lr,
                                        opt=optimizer_name,dropout_p=dropout_p,use_bn=use_bn,seed=seed)},
                       save_path)

    # load best and evaluate on test
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
    print(f"test dataset: loss {test_loss:.4f} acc {test_acc:.4f}")

    # plot curves
    plt.figure()
    plt.plot(hist["train_loss"], label="train loss")
    plt.plot(hist["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.show()

    plt.figure()
    plt.plot(hist["train_acc"], label="train acc")
    plt.plot(hist["val_acc"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
    plt.show()

    return test_acc, hist

if __name__ == "__main__":
    # default run
    train(epochs=10,
        batch_size=128,
        lr=1e-3, # learning rate
        optimizer_name="adam",
        dropout_p=0.0, # set to 0.0 for "no dropout"
        use_bn=False, # flag for batch normalization
        seed=42
    )

