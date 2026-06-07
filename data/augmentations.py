import torch
from dataclasses import dataclass
import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # channel dropout
    channel_dropout_prob: float = 0.5
    channel_dropout_frac: float = 0.1

    # gaussian noise
    noise_prob: float = 0.5
    noise_std: float = 0.05

    # time-shift
    time_shift_prob: float = 0.5
    time_shift_max: int = 32

    # time masking
    time_mask_prob: float = 0.3
    time_mask_max: int = 32


class Augmentor:
    """Applies the augmentation configuration to a single window of MEG data."""
    def __init__(self, config: AugmentationConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """Apply augmentations to the input window.

        :param window: input MEG data of shape (channels, time_points).
        :return: augmented MEG data of the same shape.
        """
        cfg = self.config
        x = window.copy()

        if self.rng.random() < cfg.channel_dropout_prob:
            x = self._channel_dropout(x, cfg.channel_dropout_frac)
        if self.rng.random() < cfg.noise_prob:
            x = self._add_gaussian_noise(x, cfg.noise_std)
        if self.rng.random() < cfg.time_shift_prob:
            x = self._time_shift(x, cfg.time_shift_max)
        if self.rng.random() < cfg.time_mask_prob:
            x = self._time_mask(x, cfg.time_mask_max)

        return x

    def _channel_dropout(self, x: np.ndarray, frac: float) -> np.ndarray:
        """Randomly drop a fraction of channels."""
        n_drop = max(1, int(x.shape[0] * frac))
        idx = self.rng.choice(x.shape[0], n_drop, replace=False)
        x[idx, :] = 0.0
        return x

    def _add_gaussian_noise(self, x: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to the data."""
        return x + self.rng.normal(0.0, std, size=x.shape).astype(np.float32)

    def _time_shift(self, x: np.ndarray, max_shift: int) -> np.ndarray:
        """Randomly shift the data in time."""
        shift = int(self.rng.integers(-max_shift, max_shift + 1))
        return np.roll(x, shift, axis=1)

    def _time_mask(self, x: np.ndarray, max_len: int) -> np.ndarray:
        """Randomly mask a segment of the time dimension from the window."""
        max_len = int(self.rng.integers(1, max_len + 1))
        start = int(self.rng.integers(0, x.shape[1] - max_len + 1))
        x[:, start:start + max_len] = 0.0
        return x


def mixup_batch(
        X: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup augmentation to a batch of data.

    :param X: input data of shape (batch_size, channels, time_points).
    :param y: labels of shape (batch_size,).
    :return: (mixed_X, y_a, y_b, lam)
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam
