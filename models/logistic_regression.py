import numpy as np
from sklearn.preprocessing import StandardScaler
from .base_model import AbstractModel
from sklearn.linear_model import LogisticRegression

"""Logistic Regression model implementation"""


def _summary_features(window: np.ndarray) -> np.ndarray:
    """Obtain summary features (mean, std, min, max) from a window of data.

    :param window: input data window of shape (n_channels, time)
    :return: summary features of shape (n_channels * 4,)
    """
    return np.concatenate([
        window.mean(axis=1), window.std(axis=1),
        window.min(axis=1),  window.max(axis=1),
    ])


def _loader_to_features(loader) -> tuple[np.ndarray, np.ndarray]:
    """Convert a DataLoader into features and labels.
    Iterates over all batches and extracts summary features.

    :param loader: dataLoader for batches (X_batch, y_batch) pairs
    :return: tuple (X,y) of features (X) and labels (y)
    """
    feats, labels = [], []
    for X_batch, y_batch in loader:
        X_np = X_batch.numpy()
        for i in range(len(X_np)):
            feats.append(_summary_features(X_np[i]))
            labels.append(y_batch[i].item())
    return np.array(feats), np.array(labels)


class LogisticRegressionModel(AbstractModel):
    def __init__(self) -> None:
        """Initialize the Logistic Regression model"""
        self.model = None
        self.scaler = None
        self.create()

    def create(self) -> tuple:
        """Create Logistic Regression model

        :return: model and scaler created
        """
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        return self.model, self.scaler

    def fit(self, train_loader, val_loader=None) -> None:
        """Train the Logistic Regression model

        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        """
        if self.model is None or self.scaler is None:
            raise ValueError(
                "Logistic Regression model has not been created yet"
            )

        X_train, y_train = _loader_to_features(train_loader)
        X_train = self.scaler.fit_transform(X_train)

        self.model.fit(X_train, y_train)
        history = {
            'train_accuracy': self.model.score(X_train, y_train)
        }

        if val_loader is not None:
            X_val, y_val = _loader_to_features(val_loader)
            history['val_accuracy'] = self.model.score(
                self.scaler.transform(X_val), y_val
            )

        return history

    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model

        :param loader: DataLoader for input features
        :return: predicted values
        """
        if self.model is None or self.scaler is None:
            raise ValueError(
                "Logistic Regression model has not been created yet"
            )
        X, y = _loader_to_features(loader)

        return self.model.predict(self.scaler.transform(X)), y
