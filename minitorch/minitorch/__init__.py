"""
MiniTorch — A minimal deep learning framework for educational purposes.

Built from scratch with only NumPy as a dependency, MiniTorch demonstrates
the core concepts behind PyTorch / TensorFlow:
    - Reverse-mode automatic differentiation (autograd)
    - Modular neural network layers (nn.Module)
    - Optimizers (SGD, Adam)
    - Loss functions (MSE, CrossEntropy)

Example usage
-------------
>>> import minitorch as mt
>>> model = mt.nn.Sequential(
...     mt.nn.Linear(784, 128),
...     mt.nn.ReLU(),
...     mt.nn.Linear(128, 10),
... )
>>> optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
>>> loss_fn = mt.loss.CrossEntropyLoss()
>>>
>>> x = mt.Tensor.randn(32, 784)       # batch of 32
>>> logits = model(x)
>>> loss = loss_fn(logits, target)
>>> loss.backward()
>>> optimizer.step()
>>> optimizer.zero_grad()
"""

from minitorch.tensor import Tensor
from minitorch import nn
from minitorch import optim
from minitorch import loss

__version__ = "0.1.0"
__all__ = ["Tensor", "nn", "optim", "loss"]