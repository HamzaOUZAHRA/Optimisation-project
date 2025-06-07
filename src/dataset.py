# src/dataset.py
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from config import DATA_ROOT, BATCH_SIZE

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

def get_loaders():
    train_ds = MNIST(root=DATA_ROOT, train=True,
                     download=True,  # <-- avant c’était False
                     transform=_transform)
    test_ds  = MNIST(root=DATA_ROOT, train=False,
                     download=True,  # <-- avant c’était False
                     transform=_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

if __name__ == "__main__":
    tr, te = get_loaders()
    print("Train size:", len(tr.dataset), " Test size:", len(te.dataset))
    x_batch, y_batch = next(iter(tr))
    print("Batch shape:", x_batch.shape, y_batch.shape)
