import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def get_mnist():
    with np.load('./data/mnist.npz') as data:
        x_train_full = data['x_train']
        y_train_full = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
    x_train_full = x_train_full / 255.0
    return x_train_full, y_train_full

class MnistDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = np.reshape(y, (-1, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_idx = self.X[idx]
        y_idx = self.y[idx]
        return torch.Tensor(X_idx), torch.LongTensor(y_idx)

def get_train_dataloader(args):
    x_train_full, y_train_full = get_mnist()
    x_train, y_train = x_train_full[5000:], y_train_full[5000:]
    train_dataset = MnistDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader


def get_val_dataloader(args):
    x_train_full, y_train_full = get_mnist()
    x_val, y_val = x_train_full[:5000], y_train_full[:5000]
    val_dataset = MnistDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return val_dataloader