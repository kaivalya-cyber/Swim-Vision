# This file defines the CNN-plus-LSTM phase classifier used by SwimVision training.
"""Phase-classifier model definition for SwimVision."""

from __future__ import annotations

import argparse
import logging

import torch
from torch import nn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


class PhaseClassifier(nn.Module):
    """CNN-plus-LSTM model that predicts swim-start phases per frame.

    Args:
        None.
    """

    def __init__(self) -> None:
        """Initialize the frame encoder, recurrent backbone, and output head.

        Args:
            None.

        Returns:
            None.
        """

        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(66, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.output_head = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass on a batch of keypoint sequences.

        Args:
            x: Input tensor with shape ``[batch, T, 33, 2]``.

        Returns:
            Logits tensor with shape ``[batch, T, 4]``.
        """

        try:
            if x.ndim != 4 or x.shape[-2:] != (33, 2):
                raise ValueError(f"Expected input shape [batch, T, 33, 2], got {tuple(x.shape)}.")
            batch_size, sequence_length, _, _ = x.shape
            flattened = x.reshape(batch_size, sequence_length, 66)
            encoded = self.frame_encoder(flattened)
            recurrent_output, _ = self.lstm(encoded)
            logits = self.output_head(recurrent_output)
            return logits
        except Exception as exc:
            raise RuntimeError(f"PhaseClassifier forward pass failed: {exc}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for classifier smoke testing.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Smoke-test the SwimVision phase classifier.")
    parser.add_argument("--batch_size", type=int, default=2, help="Dummy batch size for the smoke test.")
    parser.add_argument("--sequence_length", type=int, default=90, help="Dummy sequence length.")
    return parser


def main() -> int:
    """Run the classifier smoke-test CLI.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    model = PhaseClassifier()
    dummy_input = torch.zeros((args.batch_size, args.sequence_length, 33, 2), dtype=torch.float32)

    try:
        logits = model(dummy_input)
    except Exception as exc:
        LOGGER.error("Classifier smoke test failed: %s", exc)
        return 1

    LOGGER.info("Classifier output shape: %s", tuple(logits.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
