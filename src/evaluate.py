import torch
import torch.nn as nn
import torch.optim as optim


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.float()
            y = y.float().permute(0,3,1,2)  # (N,H,W,1) -> (N,1,H,W)

            output = model(x)  # (N,1,H,W)
            output = torch.sigmoid(output) > 0.5  # binary threshold

            all_preds.append(output)
            all_labels.append(y)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0).int()
    all_labels = torch.cat(all_labels, dim=0).int()

    # Flatten to 1D arrays for sklearn
    all_preds = all_preds.cpu().numpy().reshape(-1)
    all_labels = all_labels.cpu().numpy().reshape(-1)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
