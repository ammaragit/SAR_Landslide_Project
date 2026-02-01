import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # 4-channel SAR input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample by 2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # Downsample by 2 again
        )

        # Classifier (1 channel output for binary segmentation)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)          # Extract features
        x = self.classifier(x)        # Produce raw logits
        x = F.interpolate(x, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear', align_corners=False)  
        # Upsample back to original input size (optional)
        return x
