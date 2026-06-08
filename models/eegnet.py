import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from .base_model import AbstractModel


class EEGNet(nn.Module):
    def __init__(self, n_channels: int = 248, n_classes: int = 4, dropout: float = 0.5,
                 window_size: int = 256, F1: int = 8, D: int = 2, F2: int = 16) -> None:
        """Initialise the EEGNet model.

        :param n_channels: number of input channels.
        :param n_classes: number of output classes.
        :param dropout: dropout rate.
        :param window_size: number of time points in the input window.
        :param F1: number of filters in the first convolutional layer.
        :param D: depth multiplier for the depthwise convolution.
        :param F2: number of filters in the second convolutional layer.
        """
        super().__init__()
        # block 1, temporal convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(F2 * (window_size // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: input tensor of shape (batch_size, n_channels, window_size).
        :return: output tensor of shape (batch_size, n_classes).
        """
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class EEGNetModel(AbstractModel):
    def __init__(self, n_channels=248, n_classes=4, window_size=256, F1=8, D=2, F2=16, dropout=0.5,
                 lr=1e-3, epochs=50, device: str | None = None) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.epochs = epochs
        self.n_classes = n_classes

        self.net = EEGNet(n_channels, n_classes, dropout, window_size, F1, D, F2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            train_loss, correct, total = 0.0, 0, 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)

            val_loss, val_acc = self._evaluate(val_loader)

            history['train_loss'].append(train_loss/total)
            history['train_acc'].append(correct/total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"epoch {epoch:3d}/{self.epochs} | "
                f"train loss {train_loss/total:.4f}  acc {correct/total:.4f} | "
                f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
            )
        return history

    def predict(self, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        #all_labels, all_probs = [], []
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                #probs = torch.softmax(self.net(X), dim=1)
                logits = self.net(X)
                #all_probs.append(probs.cpu().numpy())
                #all_labels.append(probs.argmax(1).cpu().numpy())
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        #return np.concatenate(all_labels), np.concatenate(all_probs)
        return np.concatenate(all_preds), np.concatenate(all_labels)

    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Shared val/test loop — returns (avg_loss, accuracy)."""
        self.net.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.net(X)
                total_loss += self.criterion(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)

        return total_loss / total, correct / total
