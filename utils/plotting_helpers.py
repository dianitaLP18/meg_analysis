import matplotlib.pyplot as plt
import numpy as np
from data.process_data import downsample, normalise


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
