"""
Loss functions for training neural networks.

Each loss function is callable:
    loss_val = loss_fn(predictions, targets)

Supported losses
----------------
- MSELoss          — Mean Squared Error
- CrossEntropyLoss — combines LogSoftmax + NLLLoss

Example
-------
>>> loss_fn = CrossEntropyLoss()
>>> logits = model(x)            # (batch, num_classes)
>>> loss = loss_fn(logits, targets)   # targets: (batch,) integer labels
>>> loss.backward()
"""

from __future__ import annotations

import numpy as np
from typing import Union

from minitorch.tensor import Tensor


class MSELoss:
    r"""Mean Squared Error loss.

    .. math::

        \mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2

    Parameters
    ----------
    reduction : str, default "mean"
        Either ``"mean"`` (average over batch) or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction

    def __call__(self, pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        """Compute MSE between predictions and targets.

        Parameters
        ----------
        pred : Tensor
            Predictions of shape ``(N, ...)``.
        target : Tensor or np.ndarray
            Targets of the same shape as ``pred``.
        """
        if not isinstance(target, Tensor):
            target = Tensor(target, requires_grad=False)

        diff = pred - target
        sq = diff * diff

        if self.reduction == "mean":
            return sq.mean()
        return sq.sum()

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss:
    r"""Cross-entropy loss for multi-class classification.

    Computes:

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log\left(
            \frac{e^{x_{i,y_i}}}{\sum_j e^{x_{i,j}}}
        \right)

    This is equivalent to ``LogSoftmax`` followed by ``NLLLoss``.
    The input is expected to be **raw logits** (scores before softmax).

    Parameters
    ----------
    reduction : str, default "mean"
        Either ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction

    def __call__(
        self,
        logits: Tensor,
        targets: Union[Tensor, np.ndarray],
    ) -> Tensor:
        """Compute cross-entropy loss.

        Parameters
        ----------
        logits : Tensor
            Raw scores of shape ``(N, C)`` where *C* is the number of classes.
        targets : Tensor or np.ndarray
            Ground-truth class indices of shape ``(N,)``.
        """
        if not isinstance(targets, Tensor):
            targets_arr = np.asarray(targets, dtype=np.int64)
        else:
            targets_arr = targets.data.astype(np.int64)

        N = logits.shape[0]

        # Log-Softmax (numerically stable)
        log_probs = logits.log_softmax(axis=-1)

        # Gather log-probabilities for the correct classes
        # Build one-hot style gather manually with indexing
        indices = np.arange(N), targets_arr
        gathered = log_probs[indices]  # shape (N,), each is log P(correct class)

        # Negative log-likelihood
        loss = -gathered

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"