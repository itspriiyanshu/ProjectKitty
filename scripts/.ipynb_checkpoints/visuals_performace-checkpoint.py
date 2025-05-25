import matplotlib.pyplot as plt
import json

def visualize_performance(history_file="training_history.json"):
    # Load saved history (dictionary format with keys like 'loss', 'val_loss', etc.)
    with open(history_file, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Usage: visualize_performance("training_history.json")
