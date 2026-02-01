import numpy as np
import matplotlib.pyplot as plt

def show_samples():
    trainX = np.load("data/trainX.npy")
    trainY = np.load("data/trainY.npy")

    plt.figure(figsize=(10, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(trainX[i], cmap="gray")
        plt.title(f"Label: {trainY[i]}")
        plt.axis("off")
    plt.show()
