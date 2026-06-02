"""Unit tests for neural network layers."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minitorch.tensor import Tensor
from minitorch.nn import (
    Linear,
    ReLU,
    Sigmoid,
    Sequential,
    Flatten,
    Dropout,
    BatchNorm1d,
    Parameter,
    Module,
)


def _approx_equal(a, b, rtol=1e-5, atol=1e-7):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

def test_linear_forward_shape():
    layer = Linear(4, 3)
    x = Tensor.randn(5, 4)
    out = layer(x)
    assert out.shape == (5, 3)


def test_linear_forward_values():
    # Manually set weights to verify computation
    layer = Linear(2, 2, bias=False)
    layer.weight = Parameter(Tensor([[1.0, 0.0], [0.0, 2.0]]))
    x = Tensor([[2.0, 3.0]])
    out = layer(x)
    # [2, 3] @ [[1,0],[0,2]]^T = [2, 3] @ [[1,0],[0,2]]???
    # weight = (out, in) = [[1,0],[0,2]] → x @ w.T = [[2,3]] @ [[1,0],[0,2]] = [2*1+3*0, 2*0+3*2] = [2, 6]
    assert _approx_equal(out.numpy(), [[2.0, 6.0]])


def test_linear_with_bias():
    layer = Linear(2, 2, bias=True)
    layer.weight = Parameter(Tensor([[1.0, 0.0], [0.0, 1.0]]))
    layer.bias = Parameter(Tensor([5.0, -3.0]))
    x = Tensor([[2.0, 3.0]])
    out = layer(x)
    assert _approx_equal(out.numpy(), [[7.0, 0.0]])


def test_linear_gradient():
    layer = Linear(4, 3)
    x = Tensor.randn(5, 4)
    out = layer(x).sum()
    out.backward()
    # Gradients should exist for weight and bias
    assert layer.weight.data.grad is not None
    assert layer.weight.data.grad.shape == (3, 4)
    assert layer.bias.data.grad is not None
    assert layer.bias.data.grad.shape == (3,)
    # Input gradient
    assert x.grad is not None
    assert x.grad.shape == (5, 4)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def test_relu_module():
    relu = ReLU()
    x = Tensor([-1.0, 0.0, 2.0])
    out = relu(x)
    assert _approx_equal(out.numpy(), [0.0, 0.0, 2.0])
    out.sum().backward()
    # ReLU gradient: x > 0 → 1, x <= 0 → 0
    assert _approx_equal(x.grad, [0.0, 0.0, 1.0])


def test_sigmoid_module():
    sig = Sigmoid()
    x = Tensor([0.0])
    out = sig(x)
    assert _approx_equal(out.numpy(), [0.5])


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------

def test_sequential_forward():
    model = Sequential(Linear(3, 2), ReLU(), Linear(2, 1))
    x = Tensor.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 1)


def test_sequential_parameters():
    model = Sequential(Linear(3, 2), Linear(2, 1))
    params = list(model.parameters())
    # Linear(3,2): weight(2,3) + bias(2) → 2 params
    # Linear(2,1): weight(1,2) + bias(1) → 2 params
    # Total: 4 parameters
    assert len(params) == 4


def test_sequential_gradient():
    model = Sequential(Linear(3, 2), ReLU(), Linear(2, 1))
    x = Tensor.randn(4, 3)
    out = model(x).sum()
    out.backward()
    for param in model.parameters():
        assert param.data.grad is not None


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def test_flatten():
    flatten = Flatten()
    x = Tensor.randn(2, 3, 4, 5)
    out = flatten(x)
    assert out.shape == (2, 60)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

def test_dropout_train():
    drop = Dropout(p=0.5)
    x = Tensor.ones(1000, 100)
    out = drop(x)
    # About half should be zero
    zeros = np.sum(out.numpy() == 0)
    assert 30000 < zeros < 70000  # reasonable range for p=0.5


def test_dropout_eval():
    drop = Dropout(p=0.5)
    drop.eval()
    x = Tensor.ones(100, 10)
    out = drop(x)
    assert _approx_equal(out.numpy(), np.ones((100, 10)))


# ---------------------------------------------------------------------------
# BatchNorm1d
# ---------------------------------------------------------------------------

def test_batchnorm1d_forward():
    bn = BatchNorm1d(3)
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = bn(x)
    assert out.shape == (2, 3)
    # With default gamma=1, beta=0, the output should be standardized
    # mean of column 0 = (1+4)/2 = 2.5, var = ((1-2.5)^2+(4-2.5)^2)/2 = (2.25+2.25)/2 = 2.25
    # normalized = (1-2.5)/sqrt(2.25) = -1.5/1.5 = -1.0
    # normalized = (4-2.5)/1.5 = 1.0
    assert _approx_equal(out.numpy()[0, 0], -1.0)
    assert _approx_equal(out.numpy()[1, 0], 1.0)


# ---------------------------------------------------------------------------
# Module parameter discovery
# ---------------------------------------------------------------------------

def test_parameter_discovery_nested():
    class MyModel(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(4, 3)
            self.fc2 = Linear(3, 2)
            self.scale = Parameter(Tensor([2.0]))

    model = MyModel()
    params = list(model.parameters())
    # fc1: w(3,4)+b(3), fc2: w(2,3)+b(2), scale: (1,)
    assert len(params) == 5


# ---------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------

def test_state_dict():
    layer = Linear(2, 2, bias=True)
    sd = layer.state_dict()
    assert "weight" in sd
    assert "bias" in sd
    assert sd["weight"].shape == (2, 2)
    assert sd["bias"].shape == (2,)


def test_load_state_dict():
    layer = Linear(2, 2, bias=True)
    orig_weight = layer.weight.data.numpy().copy()
    state = {"weight": np.ones((2, 2)), "bias": np.zeros(2)}
    layer.load_state_dict(state)
    assert _approx_equal(layer.weight.data.numpy(), np.ones((2, 2)))
    assert _approx_equal(layer.bias.data.numpy(), np.zeros(2))