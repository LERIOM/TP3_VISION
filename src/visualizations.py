import matplotlib.pyplot as plt



def plot_training_history(losses, accuracys, val_losses):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracys, label="Validation Accuracy")
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("visualizations/training_history.png")  
    plt.show()

def plot_histogram_times(times):
    plt.figure(figsize=(8, 5))
    plt.hist(times, bins=10, edgecolor='black')
    plt.title("Distribution of Epoch Times")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.savefig("visualizations/epoch_times_histogram.png")  
    plt.show()