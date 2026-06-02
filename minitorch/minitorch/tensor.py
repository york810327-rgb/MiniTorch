"""
Tensor — the core data structure of MiniTorch.

A Tensor wraps a NumPy array and tracks operations in a computation graph
to enable automatic differentiation via reverse-mode autograd (backpropagation).

Design Philosophy:
    Every operation on a Tensor produces a new Tensor that records a reference
    to its "parents" (inputs) and the "backward function" needed to propagate
    gradients through that operation.  Calling .backward() on a scalar Tensor
    topologically sorts the graph and applies the chain rule.

Examples
--------
>>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
>>> c = (a * b).sum()
>>> c.backward()
>>> a.grad  # should equal b.data
>>> b.grad  # should equal a.data
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

ScalarOrArray = Union[int, float, Sequence[float], np.ndarray]


class Tensor:
    """A multi-dimensional array with automatic differentiation support.

    Parameters
    ----------
    data : scalar, sequence, or np.ndarray
        The underlying numeric data.
    requires_grad : bool, default True
        If True, operations on this Tensor will be tracked for gradient
        computation.
    dtype : np.dtype, optional
        Desired data type.  Inferred from `data` when omitted.
    """

    __slots__ = (
        "_data",
        "_requires_grad",
        "_grad",
        "_parents",
        "_backward_fn",
        "_op_name",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        data: ScalarOrArray,
        requires_grad: bool = True,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        arr = np.asarray(data, dtype=dtype or np.float32)
        self._data: np.ndarray = arr
        self._requires_grad: bool = requires_grad
        self._grad: Optional[np.ndarray] = None
        self._parents: Tuple[Tensor, ...] = ()
        self._backward_fn: Optional[Callable[[], None]] = None
        self._op_name: str = "leaf"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """Raw NumPy array backing this Tensor (read-only view)."""
        return self._data

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value

    @property
    def grad(self) -> Optional[np.ndarray]:
        """Accumulated gradient.  Populated after calling .backward()."""
        return self._grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def _register_parents(
        self, parents: Tuple[Tensor, ...], backward_fn: Callable[[], None], op_name: str
    ) -> None:
        """Record the inputs and gradient propagation function for this node."""
        self._parents = tuple(p for p in parents if p._requires_grad)
        self._backward_fn = backward_fn
        self._op_name = op_name

    def zero_grad(self) -> None:
        """Reset accumulated gradients to None."""
        self._grad = None

    # ------------------------------------------------------------------
    # Autograd engine
    # ------------------------------------------------------------------
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Compute gradients of this Tensor w.r.t. all leaves in the graph.

        This implements reverse-mode automatic differentiation by:
        1. Topologically sorting the computation graph (Kahn's algorithm).
        2. Seeding the output gradient (1.0 for scalars, or the provided `grad`).
        3. Traversing nodes in reverse topological order, applying the chain rule.

        Parameters
        ----------
        grad : np.ndarray, optional
            Upstream gradient.  Defaults to an array of ones with the same
            shape as ``self.data`` (equivalent to the derivative of the
            identity function).
        """
        if not self._requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self._data, dtype=self.dtype)
        elif grad.shape != self.shape:
            # Broadcast-friendly: sum over broadcast axes so the grad shape
            # matches the node shape.
            grad = _sum_to_shape(grad, self.shape)

        self._grad = grad.copy() if self._grad is None else self._grad + grad

        # ---------- topological sort ----------
        topo: List[Tensor] = []
        visited: Dict[int, bool] = {}
        in_degree: Dict[int, int] = defaultdict(int)

        def _build_graph(node: Tensor) -> None:
            nid = id(node)
            if nid in visited:
                return
            visited[nid] = False  # not yet processed for topo
            for p in node._parents:
                in_degree[nid] = in_degree.get(nid, 0)
                in_degree[id(p)] = in_degree.get(id(p), 0) + 1
                _build_graph(p)

        _build_graph(self)

        # --- Kahn's algorithm ---
        queue: List[Tensor] = [self]
        while queue:
            node = queue.pop(0)
            topo.append(node)
            for p in node._parents:
                pid = id(p)
                in_degree[pid] -= 1
                if in_degree[pid] == 0:
                    queue.append(p)

        # ---------- forward traversal (output → leaves) ----------
        # Gradient flows from the output node down to the leaf nodes.
        # topo is ordered from output to leaves (Kahn starting from self),
        # so we iterate in that order — no reversal needed.
        for node in topo:
            if node._backward_fn is not None and node._grad is not None:
                node._backward_fn()

    # ------------------------------------------------------------------
    # Gradient accumulation (used by backward functions)
    # ------------------------------------------------------------------
    def _acc_grad(self, grad: np.ndarray) -> None:
        """Add `grad` to this node's accumulated gradient."""
        if grad.shape != self.shape:
            grad = _sum_to_shape(grad, self.shape)
        if self._grad is None:
            self._grad = grad.copy()
        else:
            self._grad = self._grad + grad

    # ------------------------------------------------------------------
    # Factory / utility
    # ------------------------------------------------------------------
    def detach(self) -> Tensor:
        """Return a new Tensor sharing the same data but detached from the graph."""
        t = Tensor(self._data.copy(), requires_grad=False)
        return t

    @staticmethod
    def zeros(*shape: int, requires_grad: bool = True) -> Tensor:
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape: int, requires_grad: bool = True) -> Tensor:
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape: int, requires_grad: bool = True) -> Tensor:
        return Tensor(
            np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad
        )

    @staticmethod
    def uniform(
        *shape: int, low: float = 0.0, high: float = 1.0, requires_grad: bool = True
    ) -> Tensor:
        return Tensor(
            np.random.uniform(low, high, shape).astype(np.float32),
            requires_grad=requires_grad,
        )

    # ------------------------------------------------------------------
    # Operator overloads  (delegate to ops.py functions)
    # ------------------------------------------------------------------
    def __add__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _add(self, _wrap(other))

    def __radd__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _add(_wrap(other), self)

    def __sub__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _sub(self, _wrap(other))

    def __rsub__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _sub(_wrap(other), self)

    def __mul__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _mul(self, _wrap(other))

    def __rmul__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _mul(_wrap(other), self)

    def __truediv__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _truediv(self, _wrap(other))

    def __rtruediv__(self, other: Union[Tensor, ScalarOrArray]) -> Tensor:
        return _truediv(_wrap(other), self)

    def __matmul__(self, other: Tensor) -> Tensor:
        return _matmul(self, other)

    def __neg__(self) -> Tensor:
        return _neg(self)

    def __pow__(self, exponent: float) -> Tensor:
        return _pow(self, exponent)

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return _mean(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: int) -> Tensor:
        return _reshape(self, shape)

    def transpose(self, axes: Optional[Sequence[int]] = None) -> Tensor:
        return _transpose(self, axes)

    def relu(self) -> Tensor:
        return _relu(self)

    def log_softmax(self, axis: int = -1) -> Tensor:
        return _log_softmax(self, axis=axis)

    def exp(self) -> Tensor:
        return _exp(self)

    def log(self) -> Tensor:
        return _log(self)

    def __repr__(self) -> str:
        return f"Tensor({self._data}, requires_grad={self._requires_grad})"

    def __getitem__(self, idx: Any) -> Tensor:
        return _getitem(self, idx)

    # ------------------------------------------------------------------
    # NumPy interop
    # ------------------------------------------------------------------
    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self._data.copy()

    def item(self) -> float:
        """Return the value of a scalar Tensor as a Python float."""
        return float(self._data)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _wrap(value: Union[Tensor, ScalarOrArray]) -> Tensor:
    """Wrap a scalar/array into a Tensor (no grad tracking)."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)


def _sum_to_shape(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce `grad` to `target_shape` by summing over broadcast axes."""
    if grad.shape == target_shape:
        return grad
    # Sum leading dimensions if necessary
    ndiff = grad.ndim - len(target_shape)
    if ndiff > 0:
        axis = tuple(range(ndiff))
        grad = grad.sum(axis=axis)
    # Sum dimensions where target_shape == 1 but grad.shape > 1
    axis = tuple(i for i, (gs, ts) in enumerate(zip(grad.shape, target_shape)) if ts == 1 and gs > 1)
    if axis:
        grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


# ------------------------------------------------------------------
# Forward operations  (each defines its backward pass via closures)
# ------------------------------------------------------------------

def _add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition: out = a + b."""
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            if a.requires_grad:
                a._acc_grad(out._grad)  # type: ignore[arg-type]
            if b.requires_grad:
                b._acc_grad(out._grad)  # type: ignore[arg-type]

        out._register_parents((a, b), _backward, "add")

    return out


def _sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction: out = a - b."""
    out = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad)
            if b.requires_grad:
                b._acc_grad(-grad)

        out._register_parents((a, b), _backward, "sub")

    return out


def _mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication: out = a * b."""
    out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad * b.data)
            if b.requires_grad:
                b._acc_grad(grad * a.data)

        out._register_parents((a, b), _backward, "mul")

    return out


def _truediv(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division: out = a / b."""
    out = Tensor(a.data / (b.data + 1e-12), requires_grad=a.requires_grad or b.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            b_safe = b.data + 1e-12
            if a.requires_grad:
                a._acc_grad(grad / b_safe)
            if b.requires_grad:
                b._acc_grad(-grad * a.data / (b_safe * b_safe))

        out._register_parents((a, b), _backward, "truediv")

    return out


def _neg(a: Tensor) -> Tensor:
    """Negation: out = -a."""
    out = Tensor(-a.data, requires_grad=a.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(-grad)

        out._register_parents((a,), _backward, "neg")

    return out


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication: out = a @ b."""
    out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                # dL/dA = dL/dOut @ B^T
                a._acc_grad(_matmul_grad_a(grad, b.data))
            if b.requires_grad:
                # dL/dB = A^T @ dL/dOut
                b._acc_grad(_matmul_grad_b(a.data, grad))

        out._register_parents((a, b), _backward, "matmul")

    return out


def _matmul_grad_a(grad: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute grad w.r.t. A for A @ B, handling both 1D/2D/batched."""
    if grad.ndim <= 2 and b.ndim <= 2:
        return grad @ b.T
    # Simple batched matmul gradient
    return np.matmul(grad, _swap_last_two_axes(b))


def _matmul_grad_b(a: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Compute grad w.r.t. B for A @ B."""
    if a.ndim <= 2 and grad.ndim <= 2:
        return a.T @ grad
    return np.matmul(_swap_last_two_axes(a), grad)


def _swap_last_two_axes(arr: np.ndarray) -> np.ndarray:
    return np.swapaxes(arr, -1, -2)


def _pow(a: Tensor, exponent: float) -> Tensor:
    """Power: out = a ** exponent."""
    out = Tensor(a.data**exponent, requires_grad=a.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad * exponent * a.data ** (exponent - 1))

        out._register_parents((a,), _backward, "pow")

    return out


def _sum(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Sum along an axis."""
    out_data = a.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad)

    if out.requires_grad:
        out_shape_before = a.data.shape

        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                # Broadcast the gradient back to the original shape
                if axis is not None and not keepdims:
                    grad_expanded = np.expand_dims(grad, axis=axis)
                else:
                    grad_expanded = grad
                a._acc_grad(np.broadcast_to(grad_expanded, out_shape_before))

        out._register_parents((a,), _backward, "sum")

    return out


def _mean(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Mean along an axis."""
    n = a.data.size if axis is None else a.data.shape[axis]
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)

    if out.requires_grad:
        orig_shape = a.data.shape

        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                if axis is not None and not keepdims:
                    grad_expanded = np.expand_dims(grad, axis=axis)
                else:
                    grad_expanded = grad
                a._acc_grad(np.broadcast_to(grad_expanded, orig_shape) / n)

        out._register_parents((a,), _backward, "mean")

    return out


def _reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape preserving the underlying data."""
    out = Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)

    if out.requires_grad:
        orig_shape = a.data.shape

        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad.reshape(orig_shape))

        out._register_parents((a,), _backward, "reshape")

    return out


def _transpose(a: Tensor, axes: Optional[Sequence[int]] = None) -> Tensor:
    """Transpose / permute axes."""
    out = Tensor(a.data.transpose(axes), requires_grad=a.requires_grad)

    if out.requires_grad:
        if axes is None:
            rev_axes = None
        else:
            rev_axes = np.argsort(axes)

        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad.transpose(rev_axes))

        out._register_parents((a,), _backward, "transpose")

    return out


def _relu(a: Tensor) -> Tensor:
    """Rectified Linear Unit: out = max(0, a)."""
    mask = (a.data > 0).astype(a.dtype)
    out = Tensor(a.data * mask, requires_grad=a.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad * mask)

        out._register_parents((a,), _backward, "relu")

    return out


def _exp(a: Tensor) -> Tensor:
    """Exponential: out = e^a."""
    out_data = np.exp(a.data)
    out = Tensor(out_data, requires_grad=a.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad * out_data)

        out._register_parents((a,), _backward, "exp")

    return out


def _log(a: Tensor) -> Tensor:
    """Natural logarithm: out = log(a)."""
    out = Tensor(np.log(a.data + 1e-12), requires_grad=a.requires_grad)

    if out.requires_grad:
        a_safe_inv = 1.0 / (a.data + 1e-12)

        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                a._acc_grad(grad * a_safe_inv)

        out._register_parents((a,), _backward, "log")

    return out


def _log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Log-Softmax along `axis` (numerically stable)."""
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True) + 1e-12)
    out = Tensor(shifted - log_sum_exp, requires_grad=a.requires_grad)

    if out.requires_grad:
        softmax = np.exp(out.data)  # shape same as input
        orig_shape = a.data.shape
        out_shape = out.data.shape

        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                # d(log_softmax)/dx = I - softmax (broadcasting over axis)
                # but grad may need sum reduction if output shape differs
                g = grad.copy()
                if g.shape != softmax.shape:
                    g = np.broadcast_to(g, softmax.shape)
                jac = g - softmax * g.sum(axis=axis, keepdims=True)
                if jac.shape != orig_shape:
                    jac = _sum_to_shape(jac, orig_shape)
                a._acc_grad(jac)

        out._register_parents((a,), _backward, "log_softmax")

    return out


def _getitem(a: Tensor, idx: Any) -> Tensor:
    """Indexing / slicing."""
    out = Tensor(a.data[idx], requires_grad=a.requires_grad)

    if out.requires_grad:
        def _backward() -> None:
            grad = out._grad  # type: ignore[union-attr]
            if a.requires_grad:
                full_grad = np.zeros_like(a.data)
                full_grad[idx] = grad
                a._acc_grad(full_grad)

        out._register_parents((a,), _backward, "getitem")

    return out