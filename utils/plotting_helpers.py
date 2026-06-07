import matplotlib.pyplot as plt
import numpy as np
import os
from data.process_data import downsample, normalise
import seaborn as sns
from sklearn.metrics import confusion_matrix


# root for saving plots
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def compare_normalisations(raw_matrix: np.ndarray, save_path: str,
                           channel: int = 0, n_timesteps: int = 1000) -> None:
    """Plots the raw, z-score, and min-max normalised versions of a single channel for comparision.

    :param raw_matrix: the original raw matrix to compare.
    :param save_path: the path to save the resulting plot.
    :param channel: the channel index to plot.
    :param n_timesteps: the number of timesteps to plot for each version.
    """
    downsampled = downsample(raw_matrix)
    z = normalise(downsampled, method='zscore')
    min_max = normalise(downsampled, method='minmax')

    versions = [('Raw (downsampled)', downsampled), ('Z-score Normalisation', z),
                ('Min-Max Normalisation', min_max)]

    fig, axes = plt.subplots(len(versions), 3, figsize=(15, 10))

    # histograms of all values
    for ax, (name, data) in zip(axes[0], versions):
        ax.hist(data.flatten(), bins=100)
        ax.set_yscale('log')
        ax.set_title(f'{name}\nvalue distribution')
        ax.set_xlabel('value')
        ax.set_ylabel('log(count)')

    # one channel over time
    for ax, (name, data) in zip(axes[1], versions):
        ax.plot(data[channel, :n_timesteps])
        ax.set_title(f'{name}\nchannel {channel}, first {n_timesteps} steps')
        ax.set_xlabel('time step')
        ax.set_ylabel('value')

    # boxplots across channels
    sample_channels = np.linspace(0, data.shape[0] - 1, 20, dtype=int)
    for ax, (name, data) in zip(axes[2], versions):
        ax.boxplot([data[c] for c in sample_channels], showfliers=True)
        ax.set_title(f'{name}\nper-channel spread (20 sample channels)')
        ax.set_xlabel('channel index')
        ax.set_ylabel('value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_learning_curves(train_losses: list[float], val_losses: list[float], save_path: str) -> None:
    """Plots the training and validation loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title('Learning Curves (Model Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results", "final_images", save_path), dpi=120)
    plt.show()


def plot_accuracy_curves(train_accs: list[float], val_accs: list[float], save_path: str) -> None:
    """Plots the training and validation accuracy over epochs."""
    plt.figure(figsize=(8, 5))

    plt.plot(train_accs, 'o-', label='Training Accuracy', color='green', linestyle='-')
    plt.plot(val_accs, 'o--', label='Validation Accuracy', color='orange', linestyle='--')

    plt.title('Model Accuracy Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (e.g., 0.0 - 1.0)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results", "final_images", save_path), dpi=120)
    plt.show()


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], save_path: str) -> None:
    """Generates and plots a heatmap confusion matrix for model predictions."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    # Create a heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Evaluation Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results", "final_images", save_path), dpi=120)
    plt.show()
