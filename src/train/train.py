# This file trains the SwimVision phase classifier with W&B logging and early stopping.
"""Training loop for the SwimVision phase-classification model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import wandb
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train.classifier import PhaseClassifier
from src.train.dataset import SwimStartDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _select_device() -> torch.device:
    """Select the preferred training device for Apple Silicon or CPU fallback.

    Args:
        None.

    Returns:
        Torch device object.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_dataloaders(dataset: SwimStartDataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation loaders using an 80/20 split.

    Args:
        dataset: Initialized SwimStartDataset.
        batch_size: Batch size for both loaders.

    Returns:
        Train and validation DataLoaders.
    """

    labels = [int(dataset.records[index][1]) for index in range(len(dataset.records))]
    indices = list(range(len(dataset.records)))
    stratify = labels if len(set(labels)) > 1 and min(labels.count(label) for label in set(labels)) >= 2 else None
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=stratify,
    )
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _compute_loss_and_accuracy(
    model: PhaseClassifier,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Run a forward pass and compute sequence-wise loss and accuracy.

    Args:
        model: Phase classifier model.
        batch: Tuple of sequence tensors and clip labels.
        criterion: Loss function.
        device: Active training device.

    Returns:
        Loss tensor and scalar batch accuracy.
    """

    sequences, labels = batch
    sequences = sequences.to(device)
    labels = labels.to(device)

    try:
        logits = model(sequences)
    except Exception as exc:
        raise RuntimeError(f"Model forward pass failed during training: {exc}") from exc

    repeated_labels = labels.unsqueeze(1).repeat(1, logits.shape[1])
    loss = criterion(logits.reshape(-1, logits.shape[-1]), repeated_labels.reshape(-1))
    predictions = torch.argmax(logits, dim=-1)
    accuracy = float((predictions == repeated_labels).float().mean().item())
    return loss, accuracy


def train_model(
    processed_dir: str = "data/processed",
    labels_csv: str = "data/labels.csv",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> Dict[str, float]:
    """Train the SwimVision phase classifier and save the best checkpoint.

    Args:
        processed_dir: Directory of processed keypoint files.
        labels_csv: Metadata CSV path.
        epochs: Maximum number of training epochs.
        batch_size: Batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        patience: Early-stopping patience in epochs.

    Returns:
        Dictionary of best validation metrics.
    """

    device = _select_device()
    LOGGER.info("Training on device: %s", device)

    dataset = SwimStartDataset(processed_dir=processed_dir, labels_csv=labels_csv)
    train_loader, val_loader = _make_dataloaders(dataset, batch_size=batch_size)

    model = PhaseClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "best_classifier.pt"

    try:
        run = wandb.init(
            project="swimvision",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "patience": patience,
                "device": str(device),
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Weights & Biases logging: {exc}") from exc

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    try:
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            total_train_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()
                loss, _ = _compute_loss_and_accuracy(model, batch, criterion, device)
                try:
                    loss.backward()
                except Exception as exc:
                    raise RuntimeError(f"Backward pass failed: {exc}") from exc
                try:
                    optimizer.step()
                except Exception as exc:
                    raise RuntimeError(f"Optimizer step failed: {exc}") from exc
                total_train_loss += float(loss.item())
                total_train_batches += 1

            model.eval()
            total_val_loss = 0.0
            total_val_accuracy = 0.0
            total_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    loss, accuracy = _compute_loss_and_accuracy(model, batch, criterion, device)
                    total_val_loss += float(loss.item())
                    total_val_accuracy += accuracy
                    total_val_batches += 1

            train_loss = total_train_loss / max(total_train_batches, 1)
            val_loss = total_val_loss / max(total_val_batches, 1)
            val_accuracy = total_val_accuracy / max(total_val_batches, 1)
            LOGGER.info(
                "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | val_accuracy=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_accuracy,
            )

            try:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    }
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to log metrics to Weights & Biases: {exc}") from exc

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                try:
                    torch.save(model.state_dict(), checkpoint_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to save checkpoint to '{checkpoint_path}': {exc}") from exc
                LOGGER.info("Saved new best checkpoint to %s", checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping triggered after %s epochs without improvement.", patience)
                break
    finally:
        try:
            run.finish()
        except Exception as exc:
            LOGGER.warning("Failed to finish Weights & Biases run cleanly: %s", exc)

    return {"best_val_loss": best_val_loss, "best_val_accuracy": best_val_accuracy}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for model training.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Train the SwimVision phase classifier.")
    parser.add_argument("--processed_dir", default="data/processed", help="Processed keypoint directory.")
    parser.add_argument("--labels_csv", default="data/labels.csv", help="Metadata CSV path.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    return parser


def main() -> int:
    """Run the command-line interface for classifier training.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        metrics = train_model(
            processed_dir=args.processed_dir,
            labels_csv=args.labels_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
    except Exception as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1

    LOGGER.info("Best validation metrics: %s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
