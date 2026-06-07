import numpy as np
import copy
import torch
import torch.nn as nn
from .base_model import AbstractModel
from data.augmentations import mixup_batch
"""Convolutional Neural Network (CNN) model implementation"""


class CNNModel(AbstractModel):
    def __init__(self,
                 n_input_channels: int = 248,
                 n_classes: int = 4,
                 base_filters: int = 32,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-4,
                 epochs: int = 30,
                 patience: int = 7,
                 use_mixup: bool = False,
                 mixup_alpha: float = 0.2) -> None:
        """Initialize the CNN model

        :param n_input_channels: number of input channels.
        :param n_classes: number of output classes.
        :param base_filters: number of base filters.
        :param lr: learning rate.
        :param weight_decay: weight decay.
        :param epochs: number of epochs.
        :param patience: patience for early stopping.
        :param use_mixup: whether to use mixup augmentation.
        :param mixup_alpha: alpha parameter for mixup augmentation.
        """
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.model = self.create()

    def create(self) -> nn.Module:
        """Create CNN model"""

        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            device_type = 'cpu'
        self.device = torch.device(device_type)
        f = self.base_filters

        self.model = nn.Sequential(
            nn.Conv1d(self.n_input_channels, f, kernel_size=7, padding=3),
            nn.BatchNorm1d(f), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(f, f * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(f * 2), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(f * 2, f * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(f * 2), nn.ReLU(), nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(f * 2 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.n_classes)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.criterion = nn.CrossEntropyLoss()

        return self.model

    def fit(self, train_loader, val_loader=None) -> dict:
        if self.model is None:
            raise ValueError("CNN model has not been created yet")

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            correct_train, total_train = 0, 0

            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                if self.use_mixup:
                    X_mix, y_a, y_b, lam = mixup_batch(X, y, alpha=self.mixup_alpha)
                    out_train = self.model(X_mix)
                    loss = lam * self.criterion(out_train, y_a) + (1 - lam) * self.criterion(out_train, y_b)
                else:
                    out_train = self.model(X)
                    loss = self.criterion(out_train, y)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # Calculate training accuracy metrics
                correct_train += (out_train.argmax(1) == y).sum().item()
                total_train += y.size(0)

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
            train_acc = correct_train / total_train
            val_acc = correct / total

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            self.scheduler.step(avg_val)

            print(f"epoch {epoch+1}/{self.epochs} | "
                  f"train_loss={avg_train:.4f} | "
                  f"val_loss={avg_val:.4f} | val_acc={val_acc:.4f} ")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
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
