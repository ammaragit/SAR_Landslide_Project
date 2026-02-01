import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_data(batch_size=16, val_ratio=0.15):
    # Load numpy arrays
    trainX = np.load("data/trainX.npy")
    trainY = np.load("data/trainY.npy")
    testX  = np.load("data/testX.npy")
    testY  = np.load("data/testY.npy")

    # Convert inputs to torch tensors (N, C, H, W)
    trainX = torch.tensor(trainX, dtype=torch.float32).permute(0, 3, 1, 2)
    testX  = torch.tensor(testX,  dtype=torch.float32).permute(0, 3, 1, 2)

    # Convert masks to float (for BCEWithLogitsLoss)
    trainY = torch.tensor(trainY, dtype=torch.float32)
    testY  = torch.tensor(testY,  dtype=torch.float32)

    # Create full training dataset
    full_train_ds = TensorDataset(trainX, trainY)

    # Split train → train + validation
    val_size = int(len(full_train_ds) * val_ratio)
    train_size = len(full_train_ds) - val_size

    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(
        TensorDataset(testX, testY),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
