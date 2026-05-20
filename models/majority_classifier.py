import numpy as np
from collections import Counter

"""Majority Class Predictor model implementation"""


class MajorityClassModel:
    def __init__(self) -> None:
        """Initialize the Majority Class Predictor model"""
        self.model = None

    def fit(self, y: np.ndarray) -> None:
        """Create and train model to determine the most common label

        :param y: target values
        """
        self.model = Counter(y).most_common(1)[0][0]
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model

        :param X: input features
        :return: predicted values
        """
        if self.model is None:
            raise ValueError("Majority class model has not been created")
        return np.array([self.model] * len(X))

    def accuracy(self, y: np.ndarray) -> float:
        """Calculate accuracy of the model on test data

        :param y: test labels
        :return: accuracy of the prediction
        """
        y_pred = self.predict(len(y))
        accuracy = np.mean(y_pred == np.array(y))
        return accuracy
