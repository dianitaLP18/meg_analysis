import numpy as np
import torch
import torch.nn as nn
from .base_model import AbstractModel

"""Convolutional Neural Network (CNN) model implementation"""


class CNNModel(AbstractModel):
    def __init__(self, n_channels: int, n_classes: int) -> None:
        """Initialize the CNN model

        :param n_channels: number of input channels
        :param n_classes: number of output classes
        """
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.model = self.create()

    def create(self) -> nn.Module:
        """Create CNN model"""

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=248, out_channels=self.n_channels,
                      kernel_size=7, padding=3),
            nn.BatchNorm1d(self.n_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(self.n_channels, self.n_channels * 2,
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(self.n_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(self.n_channels * 2, self.n_channels * 4,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(self.n_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.n_classes)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 20

        return self.model

    def fit(self, train_loader, val_loader=None) -> None:
        """Train the CNN model

        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :return: training history
        """
        if self.model is None:
            raise ValueError("CNN model has not been created yet")

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss, correct, total = 0.0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    out = self.model(X)
                    val_loss += self.criterion(out, y).item()
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            val_acc = correct / total

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            history['val_acc'].append(val_acc)

            print(f"epoch {epoch+1}/{self.epochs} /"
                  f"train_loss={avg_train:.4f} / "
                  f"val_loss={avg_val:.4f} / val_acc={val_acc:.4f} ")

        return history

    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model

        :param loader: DataLoader for batches to predict on
        :return:  predicted labels and probabilities
        """
        if self.model is None:
            raise ValueError("CNN model has not been created yet")

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.argmax(1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        return np.concatenate(all_preds), np.concatenate(all_labels)
