from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import DataLoader


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model and return the training history."""
        pass

    @abstractmethod
    def predict(self, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions on the test set and return predicted labels and probabilities."""
        pass
