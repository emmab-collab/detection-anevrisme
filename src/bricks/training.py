"""
Model Training

Classe pour l'entraînement de modèles 3D.
"""

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Entraînement de modèles PyTorch.

    Cette classe gère :
    - L'entraînement par epochs
    - La validation
    - La sauvegarde de checkpoints
    - Le suivi des métriques

    Parameters
    ----------
    model : nn.Module
        Modèle PyTorch à entraîner
    criterion : Callable
        Fonction de loss
    optimizer : torch.optim.Optimizer
        Optimiseur
    device : str, optional
        Device ('cuda' ou 'cpu'), par défaut 'cuda'
    checkpoint_dir : str, optional
        Dossier pour sauvegarder les checkpoints

    Examples
    --------
    >>> model = UNet3DClassifier()
    >>> trainer = Trainer(model, criterion, optimizer)
    >>> trainer.fit(train_loader, val_loader, epochs=10)
    >>> trainer.save_checkpoint('best_model.pth')
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        self.best_val_acc = 0.0
        self.current_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Entraîne le modèle pour une epoch.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader d'entraînement

        Returns
        -------
        float
            Loss moyenne de l'epoch
        """
        self.model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {self.current_epoch+1} [Train]"
        ):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader: DataLoader) -> tuple:
        """
        Valide le modèle.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader de validation

        Returns
        -------
        tuple
            (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Epoch {self.current_epoch+1} [Val]"
            ):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                # Calculer accuracy (adapter selon votre modèle)
                # Supposons que labels est (B, 14) et outputs aussi
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    # Multi-output: utiliser dernière colonne pour label binaire
                    preds = torch.sigmoid(outputs[:, -1]) > 0.5
                    targets = labels[:, -1]
                else:
                    preds = torch.sigmoid(outputs) > 0.5
                    targets = labels

                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct / total

        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ):
        """
        Pipeline d'entraînement complet.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader d'entraînement
        val_loader : DataLoader, optional
            DataLoader de validation
        epochs : int, optional
            Nombre d'epochs, par défaut 10
        """
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}", end="")

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                print(f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    if self.checkpoint_dir:
                        save_path = os.path.join(
                            self.checkpoint_dir, f"best_model_epoch{epoch+1}.pth"
                        )
                        self.save_checkpoint(save_path)
                        print(f"  -> Best model saved! (Val Acc: {val_acc:.4f})")
            else:
                print()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print("=" * 60)

    def save_checkpoint(self, path: str):
        """
        Sauvegarde un checkpoint du modèle.

        Parameters
        ----------
        path : str
            Chemin de sauvegarde
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        """
        Charge un checkpoint.

        Parameters
        ----------
        path : str
            Chemin du checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.history = checkpoint.get(
            "history", {"train_loss": [], "val_loss": [], "val_acc": []}
        )

        print(f"Checkpoint loaded from: {path}")
        print(f"Resuming from epoch {self.current_epoch+1}")

    def plot_history(self):
        """Visualise l'historique d'entraînement."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        if self.history["val_loss"]:
            axes[0].plot(self.history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        if self.history["val_acc"]:
            axes[1].plot(self.history["val_acc"], label="Val Accuracy", color="green")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Validation Accuracy")
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"Trainer("
            f"device={self.device}, "
            f"current_epoch={self.current_epoch}, "
            f"best_val_acc={self.best_val_acc:.4f})"
        )
