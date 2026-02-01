from src.dataset import load_data
from src.model import SimpleCNN
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import show_samples
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print("Loading data...")
train_loader, val_loader, test_loader = load_data()

print("Training model...")
model = SimpleCNN()
train_model(model, train_loader, val_loader, epochs=100, patience=5)

print("Evaluating model...")
evaluate_model(model, test_loader)