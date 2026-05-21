import numpy as np
from collections import Counter
from .base_model import AbstractModel

"""Majority Class Predictor model implementation"""


class MajorityClassModel(AbstractModel):
    def __init__(self) -> None:
        """Initialize the Majority Class Predictor model"""
        self.model = None

    def fit(self, train_loader, val_loader) -> None:
        """Create and train model to determine the most common label

        :param train_loader: target values
        :param val_loader: validation set
        """
        all_labels = []
        for _, y in train_loader:
            all_labels.extend(y.numpy())
        self.model = Counter(all_labels).most_common(1)[0][0]
        return self.model

    def predict(self, test_loader) -> np.ndarray:
        """Make predictions using the trained model

        :param test_loader: test set
        :return: predicted labels
        """
        if self.model is None:
            raise ValueError("Majority class model has not been created")

        all_labels = []
        for _, y in test_loader:
            all_labels.extend(y.numpy())

        n = len(all_labels)
        preds = np.array([self.model] * n)
        true_labels = np.array(all_labels)
        return preds, true_labels

    def accuracy(self, y: np.ndarray) -> float:
        """Calculate accuracy of the model on test data

        :param y: test labels
        :return: accuracy of the prediction
        """
        y_pred = self.predict(len(y))
        accuracy = np.mean(y_pred == np.array(y))
        return accuracy
