"""Unit tests for loss functions."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minitorch.tensor import Tensor
from minitorch.loss import MSELoss, CrossEntropyLoss


def _approx_equal(a, b, rtol=1e-5, atol=1e-7):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# MSE Loss
# ---------------------------------------------------------------------------

def test_mse_forward():
    loss_fn = MSELoss()
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([2.0, 2.0, 2.0])
    loss = loss_fn(pred, target)
    # MSE = mean((1-2)^2 + (2-2)^2 + (3-2)^2) = mean(1+0+1) = 2/3
    assert _approx_equal(loss.item(), 2.0 / 3.0)


def test_mse_sum_reduction():
    loss_fn = MSELoss(reduction="sum")
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([2.0, 2.0, 2.0])
    loss = loss_fn(pred, target)
    assert _approx_equal(loss.item(), 2.0)


def test_mse_backward():
    loss_fn = MSELoss()
    pred = Tensor([1.0, 3.0], requires_grad=True)
    target = Tensor([2.0, 2.0], requires_grad=False)
    loss = loss_fn(pred, target)
    loss.backward()
    # Grad = 2*(pred-target)/N = 2*[-1,1]/2 = [-1, 1]
    assert _approx_equal(pred.grad, [-1.0, 1.0])


# ---------------------------------------------------------------------------
# CrossEntropy Loss
# ---------------------------------------------------------------------------

def test_ce_perfect_prediction():
    loss_fn = CrossEntropyLoss()
    # Very high logits for correct class → loss should be near 0
    logits = Tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    targets = np.array([0, 1], dtype=np.int64)
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01


def test_ce_uniform_logits():
    loss_fn = CrossEntropyLoss()
    # All logits equal → prob = 1/C → loss = -log(1/C) = log(C)
    logits = Tensor([[0.0, 0.0, 0.0]])
    targets = np.array([0], dtype=np.int64)
    loss = loss_fn(logits, targets)
    expected = np.log(3.0)  # 3 classes
    assert _approx_equal(loss.item(), expected, rtol=1e-4)


def test_ce_backward():
    loss_fn = CrossEntropyLoss()
    logits = Tensor([[1.0, 2.0]], requires_grad=True)
    targets = np.array([0], dtype=np.int64)
    loss = loss_fn(logits, targets)
    loss.backward()
    # dCE/dx_0 = softmax(x)[0] - 1, dCE/dx_1 = softmax(x)[1]
    softmax = np.exp([1, 2]) / np.sum(np.exp([1, 2]))
    expected_grad = softmax.copy()
    expected_grad[0] -= 1.0
    expected_grad /= 1.0  # mean reduction, batch size=1
    assert _approx_equal(logits.grad, expected_grad, rtol=1e-5)


def test_ce_sum_reduction():
    loss_fn = CrossEntropyLoss(reduction="sum")
    logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    targets = np.array([0, 1], dtype=np.int64)
    loss = loss_fn(logits, targets)
    # Each sample: -log(1/3) = log(3); sum = 2*log(3)
    expected = 2.0 * np.log(3.0)
    assert _approx_equal(loss.item(), expected, rtol=1e-4)


def test_ce_batch_mean():
    loss_fn = CrossEntropyLoss(reduction="mean")
    logits = Tensor([[5.0, 0.0], [0.0, 5.0]])
    targets = np.array([0, 1], dtype=np.int64)
    loss = loss_fn(logits, targets)
    # Both are very confident correct → loss near 0
    assert loss.item() < 0.02