import torch
import torch.nn as nn
import torch.optim as optim
from src.early_stopping import EarlyStopping

def train_model(model, train_loader, val_loader, epochs=100, patience=5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.float()
            y = y.permute(0, 3, 1, 2).float()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.float()
                y = y.permute(0, 3, 1, 2).float()

                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ---------- EARLY STOPPING ----------
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), "models/sar_cnn.pth")
    print("✅ Model saved: models/sar_cnn.pth")
