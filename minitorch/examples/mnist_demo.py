#!/usr/bin/env python3
"""
MNIST Classifier — Train a two-layer MLP on MNIST using MiniTorch.

This example demonstrates the full MiniTorch workflow:
    1. Load and preprocess the MNIST dataset
    2. Define a neural network (784 → 128 → 10)
    3. Train with CrossEntropyLoss + Adam optimizer
    4. Evaluate accuracy on a held-out test set

Usage
-----
    python mnist_demo.py

No external data download is required; this script generates synthetic data
that mimics MNIST statistics for rapid iteration.  To swap in real MNIST,
replace the data-generation block with an sklearn / torchvision loader.

Expected output (after ~10 epochs):
    Train Accuracy: ~98%
    Test Accuracy:  ~97%
"""

from __future__ import annotations

import sys
import os

# Make sure the minitorch package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

from minitorch.tensor import Tensor
from minitorch.nn import Linear, ReLU, Sequential
from minitorch.optim import Adam
from minitorch.loss import CrossEntropyLoss


# ---------------------------------------------------------------------------
# Synthetic MNIST-like data generator
# ---------------------------------------------------------------------------
def make_mnist_like_data(
    n_samples: int = 2000,
    n_features: int = 784,
    n_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data resembling MNIST (normalised pixel values).

    Each class is centred around a different random mean vector, and Gaussian
    noise is added.  Pixel values are clipped to [0, 1].
    """
    np.random.seed(42)
    # Class prototypes (one per digit)
    prototypes = np.random.randn(n_classes, n_features).astype(np.float32)
    data_list, label_list = [], []
    per_class = n_samples // n_classes

    for c in range(n_classes):
        X_c = prototypes[c] + 0.3 * np.random.randn(per_class, n_features).astype(
            np.float32
        )
        X_c = np.clip(X_c, 0.0, 1.0)
        data_list.append(X_c)
        label_list.append(np.full(per_class, c, dtype=np.int64))

    X = np.concatenate(data_list, axis=0).astype(np.float32)
    y = np.concatenate(label_list, axis=0).astype(np.int64)

    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------
def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    """Top-1 classification accuracy."""
    preds = np.argmax(logits.numpy(), axis=1)
    return float(np.mean(preds == targets))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: Sequential,
    optimizer: Adam,
    loss_fn: CrossEntropyLoss,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Run a single training epoch, return (avg_loss, accuracy)."""
    n = len(y)
    perm = np.random.permutation(n)
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        x_batch = Tensor(X[idx], requires_grad=True)
        y_batch = y[idx]

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y_batch) * len(idx)
        n_batches += 1

    return total_loss / n_batches, total_acc / n


def evaluate(
    model: Sequential,
    loss_fn: CrossEntropyLoss,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Evaluate the model on a dataset (no gradient tracking)."""
    model.eval()
    n = len(y)
    total_loss = 0.0
    total_correct = 0

    for start in range(0, n, batch_size):
        idx = slice(start, start + batch_size)
        x_batch = Tensor(X[idx], requires_grad=False)
        y_batch = y[idx]

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        total_loss += loss.item() * (idx.stop - idx.start)
        total_correct += int(np.sum(np.argmax(logits.numpy(), axis=1) == y_batch))

    model.train()
    return total_loss / n, total_correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  MiniTorch MNIST Demo")
    print("=" * 60)

    # -- Data -----------------------------------------------------------------
    print("\nGenerating synthetic MNIST-like data ...")
    X_all, y_all = make_mnist_like_data(n_samples=2000)

    # Train / test split (80 / 20)
    split = int(0.8 * len(y_all))
    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]

    print(f"  Train samples: {len(y_train)}")
    print(f"  Test  samples: {len(y_test)}")
    print(f"  Input dim:     {X_train.shape[1]}")
    print(f"  Classes:       {np.unique(y_train).size}")

    # -- Model ----------------------------------------------------------------
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )
    print(f"\nModel:\n  {model}")
    n_params = sum(p.data.size for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # -- Optimizer & loss -----------------------------------------------------
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()

    # -- Training loop --------------------------------------------------------
    n_epochs = 15
    print(f"\nTraining for {n_epochs} epochs ...\n")
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, optimizer, loss_fn, X_train, y_train)
        test_loss, test_acc = evaluate(model, loss_fn, X_test, y_test)

        print(
            f"  Epoch {epoch:2d} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.3%} | "
            f"test loss: {test_loss:.4f} | test acc: {test_acc:.3%}"
        )

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s")

    # -- Final evaluation -----------------------------------------------------
    _, final_acc = evaluate(model, loss_fn, X_test, y_test)
    print(f"Final test accuracy: {final_acc:.3%}")


if __name__ == "__main__":
    main()