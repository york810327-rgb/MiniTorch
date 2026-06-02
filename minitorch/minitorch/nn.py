"""
Neural network building blocks — PyTorch-style Module system.

Every layer inherits from ``Module``.  A ``Module`` owns a collection of
``Parameter`` objects that are automatically discovered via ``__dict__``
inspection, so optimizers can iterate over them without manual registration.

Supported layers
----------------
- Linear      — fully-connected / dense layer
- Conv2d      — 2D convolution (im2col-based)
- ReLU        — element-wise rectified linear unit
- Sigmoid     — element-wise sigmoid activation
- Tanh        — element-wise hyperbolic tangent
- Dropout     — stochastic regularization
- Flatten     — reshape input to 2D (batch, features)
- Sequential  — compose layers in order
- BatchNorm1d — 1D batch normalization (train/eval modes)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Tuple

from minitorch.tensor import Tensor


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------

class Parameter:
    """A trainable Tensor wrapper that is recognised by ``Module.parameters()``.

    Parameters
    ----------
    data : Tensor
        The underlying tensor, typically created via ``Tensor.randn(...)``.
    """

    def __init__(self, data: Tensor) -> None:
        self.data = data

    def zero_grad(self) -> None:
        self.data.zero_grad()

    def __repr__(self) -> str:
        return f"Parameter({self.data})"


# ---------------------------------------------------------------------------
# Module  (base class)
# ---------------------------------------------------------------------------

class Module:
    """Base class for all neural network components.

    Subclasses override ``forward()`` and optionally ``__init__()``.
    Parameters registered as ``self.<name> = Parameter(...)`` are
    automatically collected via ``parameters()``.
    """

    def __init__(self) -> None:
        self._training: bool = True

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    # -- Parameter discovery --
    def parameters(self) -> Iterator[Parameter]:
        """Recursively yield all ``Parameter`` instances owned by this module."""
        for _, val in self.__dict__.items():
            if isinstance(val, Parameter):
                yield val
            elif isinstance(val, Module):
                yield from val.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """Like ``parameters()`` but yields ``(name, param)`` pairs."""
        for name, val in self.__dict__.items():
            if isinstance(val, Parameter):
                yield name, val
            elif isinstance(val, Module):
                for sub_name, sub_val in val.named_parameters():
                    yield f"{name}.{sub_name}", sub_val

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    # -- Training / eval mode --
    def train(self) -> None:
        self._training = True
        for _, val in self.__dict__.items():
            if isinstance(val, Module):
                val.train()

    def eval(self) -> None:
        self._training = False
        for _, val in self.__dict__.items():
            if isinstance(val, Module):
                val.eval()

    @property
    def training(self) -> bool:
        return self._training

    # -- Parameter saving / loading --
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {name: param.data.numpy() for name, param in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        current = dict(self.named_parameters())
        for name, arr in state.items():
            if name not in current:
                raise KeyError(f"Parameter '{name}' not found in module.")
            current[name].data = Tensor(arr, requires_grad=True)


# ---------------------------------------------------------------------------
# Layer implementations
# ---------------------------------------------------------------------------

class Linear(Module):
    r"""Fully-connected layer:  :math:`y = xW^T + b`.

    Parameters
    ----------
    in_features : int
        Dimensionality of each input sample.
    out_features : int
        Dimensionality of each output sample.
    bias : bool, default True
        If True, an additive bias term is learned.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # Kaiming / He uniform initialisation
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(
            Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        )
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, in_features)  →  (batch, out_features)
        # weight: (out_features, in_features)
        # We need x @ weight.T
        out = x @ self.weight.data.transpose()
        if self.bias is not None:
            out = out + self.bias.data
        return out

    def extra_repr(self) -> str:
        in_f = self.weight.data.shape[1]
        out_f = self.weight.data.shape[0]
        return f"in_features={in_f}, out_features={out_f}, bias={self.bias is not None}"

    def __repr__(self) -> str:
        return f"Linear({self.extra_repr()})"


class Conv2d(Module):
    """2D convolution layer implemented via im2col + matrix multiply.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of filters / output channels.
    kernel_size : int or (int, int)
        Height and width of the convolution kernel.
    stride : int, default 1
    padding : int, default 0
    bias : bool, default True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = kernel_size
        # Kaiming uniform
        fan_in = in_channels * kh * kw
        bound = np.sqrt(6.0 / fan_in)
        self.weight = Parameter(
            Tensor.uniform(out_channels, in_channels, kh, kw, low=-bound, high=bound)
        )
        if bias:
            self.bias = Parameter(Tensor.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape
        OC = self.out_channels
        kh, kw = self.kernel_size
        pad = self.padding
        stride = self.stride

        # Pad input
        if pad > 0:
            x_padded = np.pad(
                x.data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant"
            )
        else:
            x_padded = x.data

        H_out = (H + 2 * pad - kh) // stride + 1
        W_out = (W + 2 * pad - kw) // stride + 1

        # im2col: extract patches
        cols = _im2col(x_padded, kh, kw, stride)  # shape: (N, C*kh*kw, H_out*W_out)
        cols_reshaped = cols.reshape(N, C * kh * kw, H_out * W_out)

        # Reshape weights: (OC, C*kh*kw)
        w_flat = self.weight.data.reshape(OC, C * kh * kw)

        # Actually, let's do the im2col approach properly:
        # w_flat shape (OC, C*kh*kw)
        w_flat_tensor = Tensor(w_flat, requires_grad=self.weight.data.requires_grad)

        outputs = []
        for i in range(N):
            col_tensor = Tensor(cols_reshaped[i], requires_grad=x.requires_grad)
            # (OC, C*kh*kw) @ (C*kh*kw, H_out*W_out) = (OC, H_out*W_out)
            result = w_flat_tensor @ col_tensor
            outputs.append(result.data)

        out_data = np.stack(outputs, axis=0).reshape(N, OC, H_out, W_out)

        # Add bias
        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, OC, 1, 1)

        # Return a Tensor with full autograd support
        # Since we used Tensor operations (w_flat_tensor @ col_tensor), the graph is built.
        # We need to aggregate the results properly.
        # Let's sum them for the final output tensor.

        # Better: build the output as a sum of per-sample results
        # But stacking won't preserve autograd... Let's use a different approach.

        # For now, let's implement Conv2d using Tensor operations directly,
        # which is cleaner but uses more memory.

        # Actually, let's use a simpler convolution approach with unfold
        # that directly works with Tensor operations.

        out = _conv2d_forward(x, self.weight.data, self.bias, pad, stride)
        return out

    def extra_repr(self) -> str:
        return (
            f"({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding})"
        )

    def __repr__(self) -> str:
        return f"Conv2d{self.extra_repr()}"


def _im2col(
    x: np.ndarray, kh: int, kw: int, stride: int
) -> np.ndarray:
    """Convert image patches to columns (for efficient convolution)."""
    N, C, H, W = x.shape
    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1

    # Row indices
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)

    # Column indices
    j0 = np.tile(np.arange(kw), kh * C)
    j1 = stride * np.tile(np.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # Channel index
    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)

    cols = x[:, k, i, j]
    # cols shape: (N, C*kh*kw, H_out*W_out) but channels interleaved
    cols = cols.reshape(N, C * kh * kw, H_out * W_out)
    return cols


def _conv2d_forward(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Parameter],
    padding: int,
    stride: int,
) -> Tensor:
    """Pure-Tensor conv2d forward (loop over output positions)."""
    N, C, H, W = x.shape
    OC = weight.shape[0]
    kh, kw = weight.shape[2], weight.shape[3]

    # Pad input using numpy and re-wrap
    if padding > 0:
        padded_data = np.pad(
            x.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
    else:
        padded_data = x.data
    x_padded = Tensor(padded_data, requires_grad=x.requires_grad)

    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1

    # Use numpy for the heavy forward pass but register backward
    # For a pure Tensor approach, we'll build the output cell by cell.
    # This is simpler but slower — educational purpose is fine.

    # Use a vectorized numpy forward with custom backward
    out_data = np.zeros((N, OC, H_out, W_out), dtype=np.float32)
    for oh in range(H_out):
        for ow in range(W_out):
            h_start = oh * stride
            w_start = ow * stride
            patch = padded_data[:, :, h_start : h_start + kh, w_start : w_start + kw]
            # patch: (N, C, kh, kw)  weight: (OC, C, kh, kw)
            # out = sum over C, kh, kw of patch * weight
            # Using Tensor mul and sum:
            patch_t = Tensor(patch, requires_grad=x.requires_grad)
            # Expand weight for batch dim: weight shape (OC, C, kh, kw) -> (1, OC, C, kh, kw)
            # patch shape (N, C, kh, kw) -> (N, 1, C, kh, kw) -> broadcast
            w_expanded = weight.reshape(1, OC, C, kh, kw)
            p_expanded = patch_t.reshape(N, 1, C, kh, kw)
            prod = (p_expanded * w_expanded).sum(axis=(2, 3, 4))
            out_data[:, :, oh, ow] = prod.data

    out = Tensor(out_data, requires_grad=x.requires_grad or weight.requires_grad)

    # Register custom backward
    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad
            if weight.requires_grad:
                # Compute weight gradient
                w_grad = np.zeros_like(weight.data)
                for oh in range(H_out):
                    for ow in range(W_out):
                        h_start = oh * stride
                        w_start = ow * stride
                        patch = padded_data[:, :, h_start : h_start + kh, w_start : w_start + kw]
                        # grad_out: (N, OC)
                        g = grad[:, :, oh, ow]  # (N, OC)
                        # patch: (N, C, kh, kw)
                        # w_grad += sum_N g[n, :] (outer) patch[n, :, :, :]
                        w_grad += np.tensordot(g, patch, axes=([0], [0]))  # (OC, C, kh, kw)
                weight._acc_grad(w_grad)

            if x.requires_grad:
                x_grad = np.zeros_like(x.data)
                for oh in range(H_out):
                    for ow in range(W_out):
                        h_start = oh * stride
                        w_start = ow * stride
                        g = grad[:, :, oh, ow]  # (N, OC)
                        # weight: (OC, C, kh, kw)
                        # x_grad[:, :, h_start:h_start+kh, w_start:w_start+kw] += sum_OC g * weight
                        contrib = np.tensordot(g, weight.data, axes=([1], [0]))  # (N, C, kh, kw)
                        if padding > 0:
                            x_grad[:, :, h_start + padding : h_start + padding + kh,
                                   w_start + padding : w_start + padding + kw] += contrib
                        else:
                            x_grad[:, :, h_start : h_start + kh, w_start : w_start + kw] += contrib
                x._acc_grad(x_grad)

        out._register_parents((x, weight), _backward, "conv2d")

    if bias is not None:
        out = out + bias.data.reshape(1, -1, 1, 1)

    return out


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class ReLU(Module):
    """Rectified Linear Unit:  out = max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation:  out = 1 / (1 + e^{-x})."""

    def forward(self, x: Tensor) -> Tensor:
        # Use Tensor operations
        ones = Tensor(np.ones_like(x.data), requires_grad=False)
        neg_x = -x
        return ones / (ones + (-x).exp())

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent: out = tanh(x)."""
    def forward(self, x: Tensor) -> Tensor:
        e2x = (x * 2.0).exp()
        ones = Tensor(np.ones_like(x.data), requires_grad=False)
        return (e2x - ones) / (e2x + ones)

    def __repr__(self) -> str:
        return "Tanh()"


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

class Flatten(Module):
    """Flatten input to (batch_size, -1), preserving the batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        return x.reshape(batch, -1)

    def __repr__(self) -> str:
        return "Flatten()"


class Dropout(Module):
    """During training, randomly zeroes some elements with probability *p*.

    At evaluation time, the module simply returns the input unchanged.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        mask_tensor = Tensor(mask * scale, requires_grad=False)
        return x * mask_tensor

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class BatchNorm1d(Module):
    """1D batch normalization with running estimates for eval mode.

    Parameters
    ----------
    num_features : int
        Number of features (C from an input of shape (N, C)).
    eps : float, default 1e-5
        Small constant for numerical stability.
    momentum : float, default 0.1
        Momentum for updating running mean / variance.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(Tensor.ones(num_features))
        self.beta = Parameter(Tensor.zeros(num_features))
        # Running statistics (not trainable)
        self.running_mean = Tensor.zeros(num_features, requires_grad=False)
        self.running_var = Tensor.ones(num_features, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (N, C)
        if self.training:
            mean = x.mean(axis=0)
            var = ((x - mean.reshape(1, -1)) ** 2).mean(axis=0)
            # Update running stats (numpy level)
            self.running_mean = Tensor(
                (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data,
                requires_grad=False,
            )
            self.running_var = Tensor(
                (1 - self.momentum) * self.running_var.data + self.momentum * var.data,
                requires_grad=False,
            )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean.reshape(1, -1)) / (var.reshape(1, -1) + self.eps) ** 0.5
        # Scale and shift
        return self.gamma.data.reshape(1, -1) * x_norm + self.beta.data.reshape(1, -1)

    def __repr__(self) -> str:
        return f"BatchNorm1d({self.gamma.data.shape[0]}, eps={self.eps}, momentum={self.momentum})"


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class Sequential(Module):
    """A sequential container of layers, called in order.

    Parameters
    ----------
    *layers : Module
        Layers to chain together.
    """

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)
        # Register layers as attributes for parameter discovery
        for i, layer in enumerate(self.layers):
            setattr(self, f"_layer_{i}", layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        items = "\n  ".join(repr(l) for l in self.layers)
        return f"Sequential(\n  {items}\n)"