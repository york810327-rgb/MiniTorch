"""Unit tests for the core Tensor and autograd engine."""

import numpy as np
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minitorch.tensor import Tensor


def _approx_equal(a, b, rtol=1e-5, atol=1e-7):
    """Check approximate equality of two numeric values or arrays."""
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Basic construction & properties
# ---------------------------------------------------------------------------

def test_tensor_creation():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.requires_grad is True
    assert _approx_equal(t.numpy(), [1.0, 2.0, 3.0])


def test_tensor_no_grad():
    t = Tensor([1.0, 2.0], requires_grad=False)
    assert t.requires_grad is False


def test_tensor_factories():
    z = Tensor.zeros(2, 3)
    assert z.shape == (2, 3)
    assert np.all(z.numpy() == 0)

    o = Tensor.ones(4)
    assert o.shape == (4,)
    assert np.all(o.numpy() == 1)

    r = Tensor.randn(2, 2)
    assert r.shape == (2, 2)


def test_detach():
    t = Tensor([1.0, 2.0], requires_grad=True)
    d = t.detach()
    assert d.requires_grad is False
    assert _approx_equal(t.numpy(), d.numpy())


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

def test_add_forward():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a + b
    assert _approx_equal(c.numpy(), [4.0, 6.0])


def test_add_scalar():
    a = Tensor([1.0, 2.0])
    c = a + 5.0
    assert _approx_equal(c.numpy(), [6.0, 7.0])


def test_add_backward():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a + b
    c.backward()
    assert _approx_equal(a.grad, [1.0, 1.0])
    assert _approx_equal(b.grad, [1.0, 1.0])


def test_add_broadcast_backward():
    a = Tensor([1.0, 2.0, 3.0])  # (3,)
    b = Tensor([10.0])  # (1,) broadcasts
    c = a + b
    c.backward()
    # Gradient for a should be 1 per element
    assert _approx_equal(a.grad, [1, 1, 1])
    # Gradient for b should be sum of 3 ones = 3
    assert _approx_equal(b.grad, [3])


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------

def test_sub_forward():
    a = Tensor([5.0, 7.0])
    b = Tensor([2.0, 3.0])
    c = a - b
    assert _approx_equal(c.numpy(), [3.0, 4.0])


def test_sub_backward():
    a = Tensor([5.0, 7.0])
    b = Tensor([2.0, 3.0])
    c = a - b
    c.backward()
    assert _approx_equal(a.grad, [1.0, 1.0])
    assert _approx_equal(b.grad, [-1.0, -1.0])


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

def test_mul_forward():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = a * b
    assert _approx_equal(c.numpy(), [8.0, 15.0])


def test_mul_backward():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = a * b
    c.backward()
    assert _approx_equal(a.grad, [4.0, 5.0])
    assert _approx_equal(b.grad, [2.0, 3.0])


def test_mul_scalar_backward():
    a = Tensor([2.0, 3.0])
    c = a * 10.0
    c.backward()
    assert _approx_equal(a.grad, [10.0, 10.0])


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------

def test_div_forward():
    a = Tensor([6.0, 8.0])
    b = Tensor([2.0, 4.0])
    c = a / b
    assert _approx_equal(c.numpy(), [3.0, 2.0])


def test_div_backward():
    a = Tensor([6.0, 8.0])
    b = Tensor([2.0, 4.0])
    c = a / b
    c.backward()
    # d/da (a/b) = 1/b
    assert _approx_equal(a.grad, [1 / 2, 1 / 4])
    # d/db (a/b) = -a/b^2
    assert _approx_equal(b.grad, [-6 / 4, -8 / 16])


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------

def test_matmul_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    expected = np.array([[19, 22], [43, 50]])
    assert _approx_equal(c.numpy(), expected)


def test_matmul_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.0], [0.0, 2.0]])
    c = (a @ b).sum()
    c.backward()
    # d(sum(A @ B))/dA = grad_out @ B^T, with grad_out = all ones
    # [[1,1],[1,1]] @ [[2,0],[0,2]] = [[2,2],[2,2]]
    assert _approx_equal(a.grad, [[2, 2], [2, 2]])
    # d(sum(A @ B))/dB = A^T @ grad_out
    # [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    assert _approx_equal(b.grad, [[4, 4], [6, 6]])


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

def test_neg():
    a = Tensor([1.0, -2.0])
    c = -a
    assert _approx_equal(c.numpy(), [-1.0, 2.0])
    c.backward()
    assert _approx_equal(a.grad, [-1.0, -1.0])


# ---------------------------------------------------------------------------
# Sum
# ---------------------------------------------------------------------------

def test_sum_scalar():
    a = Tensor([1.0, 2.0, 3.0])
    c = a.sum()
    assert c.item() == 6.0
    c.backward()
    assert _approx_equal(a.grad, [1.0, 1.0, 1.0])


def test_sum_axis():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    c = a.sum(axis=0)
    assert _approx_equal(c.numpy(), [4.0, 6.0])
    c.backward()
    assert _approx_equal(a.grad, [[1.0, 1.0], [1.0, 1.0]])


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------

def test_mean():
    a = Tensor([2.0, 4.0, 6.0])
    c = a.mean()
    assert c.item() == 4.0
    c.backward()
    assert _approx_equal(a.grad, [1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

def test_reshape():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    c = a.reshape(4)
    assert c.shape == (4,)
    c.backward(np.array([1.0, 2.0, 3.0, 4.0]))
    assert _approx_equal(a.grad, [[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

def test_relu_forward():
    a = Tensor([-1.0, 0.0, 2.0])
    c = a.relu()
    assert _approx_equal(c.numpy(), [0.0, 0.0, 2.0])


def test_relu_backward():
    a = Tensor([-1.0, 2.0, -3.0])
    c = a.relu().sum()
    c.backward()
    assert _approx_equal(a.grad, [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Exp / Log
# ---------------------------------------------------------------------------

def test_exp():
    a = Tensor([0.0, 1.0])
    c = a.exp()
    assert _approx_equal(c.numpy(), [1.0, np.e])
    c.backward()
    assert _approx_equal(a.grad, [1.0, np.e])


def test_log():
    a = Tensor([1.0, np.e])
    c = a.log()
    assert _approx_equal(c.numpy(), [0.0, 1.0])
    c.backward()
    assert _approx_equal(a.grad, [1.0, 1 / np.e])


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def test_pow():
    a = Tensor([1.0, 2.0, 3.0])
    c = a ** 2
    assert _approx_equal(c.numpy(), [1.0, 4.0, 9.0])
    c.backward()
    assert _approx_equal(a.grad, [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# Chain rule (composition)
# ---------------------------------------------------------------------------

def test_chain_add_mul():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = Tensor(4.0)
    # f = (a + b) * c = (2+3)*4 = 20
    # df/da = c = 4, df/db = c = 4, df/dc = a+b = 5
    out = (a + b) * c
    out.backward()
    assert a.grad.item() == 4.0
    assert b.grad.item() == 4.0
    assert c.grad.item() == 5.0


def test_nested():
    # f = (a * b + c) * d
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = Tensor(4.0)
    d = Tensor(5.0)
    # f = (6 + 4)*5 = 50
    out = (a * b + c) * d
    out.backward()
    # df/da = b * d = 3*5 = 15
    # df/db = a * d = 10
    # df/dc = d  = 5
    # df/dd = a*b + c = 10
    assert _approx_equal(a.grad, 15)
    assert _approx_equal(b.grad, 10)
    assert _approx_equal(c.grad, 5)
    assert _approx_equal(d.grad, 10)


# ---------------------------------------------------------------------------
# Gradient check (finite differences)
# ---------------------------------------------------------------------------

def _grad_check(func, inputs, eps=1e-4, rtol=1e-5):
    """Verify analytic gradients against finite-difference approximations."""
    # Compute analytic grad
    for inp in inputs:
        inp.zero_grad()
    out = func(*inputs)
    out.backward()
    anal_grads = [inp.grad.copy() for inp in inputs]

    # Finite-difference for each input
    for i, inp in enumerate(inputs):
        data = inp.data.copy()
        fd_grad = np.zeros_like(data)
        it = np.nditer(data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = data[idx]

            data[idx] = old + eps
            inp._data = data
            for x in inputs:
                x.zero_grad()
            out_plus = func(*inputs).item()

            data[idx] = old - eps
            inp._data = data
            for x in inputs:
                x.zero_grad()
            out_minus = func(*inputs).item()

            fd_grad[idx] = (out_plus - out_minus) / (2 * eps)
            data[idx] = old

        inp._data = data
        assert _approx_equal(anal_grads[i], fd_grad, rtol=rtol), (
            f"Gradient check failed for input {i}!\n"
            f"Analytic:\n{anal_grads[i]}\n"
            f"FD:\n{fd_grad}"
        )


def test_gradient_check_simple():
    def f(a, b):
        return (a * b).sum()

    a = Tensor([1.5, -2.0, 3.3])
    b = Tensor([0.7, 4.1, -1.2])
    _grad_check(f, [a, b], eps=1e-3, rtol=1e-2)


def test_gradient_check_composed():
    def f(a, b, c):
        return ((a * b).relu() + c).sum()

    a = Tensor([1.0, 1.5, 3.0])
    b = Tensor([0.5, 2.0, 1.0])
    c = Tensor([0.5, 0.5, 0.5])
    _grad_check(f, [a, b, c], eps=1e-3, rtol=1e-2)


def test_gradient_check_softmax():
    def f(a):
        return a.log_softmax().sum()

    a = Tensor([0.2, 0.5, 1.0, 2.0, -0.3])
    _grad_check(f, [a], eps=1e-3, rtol=5e-2)


def test_gradient_check_matmul():
    def f(a, b):
        return (a @ b).sum()

    a = Tensor([[0.5, 1.2], [2.1, -0.7]])
    b = Tensor([[1.0, 0.3], [0.8, -1.5]])
    _grad_check(f, [a, b], eps=1e-3, rtol=1e-2)


# ---------------------------------------------------------------------------
# Topological sort edge case
# ---------------------------------------------------------------------------

def test_diamond_graph():
    """Test a diamond-shaped computation graph: d = (a+b) * (a+c)."""
    a = Tensor(2.0)
    b = Tensor(3.0)
    c_d = Tensor(4.0)
    # d = (a+b) * (a+c) = (5)*(6) = 30
    # dd/da = (a+b) + (a+c) = 5+6 = 11
    # dd/db = a+c = 6
    # dd/dc = a+b = 5
    out = (a + b) * (a + c_d)
    out.backward()
    assert _approx_equal(a.grad, 11)
    assert _approx_equal(b.grad, 6)
    assert _approx_equal(c_d.grad, 5)


def test_multiple_consumers():
    """Single tensor used in two places."""
    x = Tensor(3.0)
    y = x * 2  # 6
    z = x * 3  # 9
    out = y + z  # 15
    # dout/dx = dout/dy * dy/dx + dout/dz * dz/dx = 1*2 + 1*3 = 5
    out.backward()
    assert _approx_equal(x.grad, 5)