# MiniTorch 🔥

<p align="center">
  <b>A minimal deep learning framework built from scratch with only NumPy.</b>
</p>

MiniTorch demonstrates the foundational concepts behind PyTorch / TensorFlow —
**reverse-mode automatic differentiation**, a **modular neural network layer
system**, **optimizers**, and **loss functions** — all implemented in pure
Python with a single dependency (NumPy).

> ⚡ **Interview-ready** — every line is documented; every gradient is checked.

---

## ✨ Features

| Module          | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| `Tensor`        | N-dimensional array with full autograd engine (topological sort + chain rule) |
| `nn`            | PyTorch-style `Module` system: Linear, Conv2d, ReLU, Sigmoid, Tanh, Dropout, BatchNorm1d, Sequential |
| `optim`         | SGD (with momentum / weight decay) and Adam                        |
| `loss`          | MSE and CrossEntropy loss functions                                |
| `tests/`        | 59 unit tests including gradient checks via finite differences      |
| `examples/`     | End-to-end MNIST classifier training demo                          |

---

## 🧠 Architecture Highlight — Autograd Engine

The core of MiniTorch is a **dynamically constructed computation graph**:

1. Every operation (`+`, `*`, `@`, `.relu()`, `.sum()`) produces a new `Tensor`
   that records its inputs (parents) and a **backward closure** encoding the
   chain rule.
2. Calling `.backward()` topologically sorts the graph via **Kahn's algorithm**
   and traverses it to accumulate gradients.
3. Gradients are computed analytically — no numerical approximation.

```
  Forward:  x = Tensor(2.0);  y = x * 3;  z = y.relu();  z.backward()
  Backward: dloss/dx = dloss/dz · dz/dy · dy/dx  (chain rule)
```

**Code snippet** (from `minitorch/tensor.py`):
```python
def _mul(a, b):
    out = Tensor(a.data * b.data, requires_grad=...)
    def _backward():
        a._acc_grad(out._grad * b.data)
        b._acc_grad(out._grad * a.data)
    out._register_parents((a, b), _backward, "mul")
    return out
```

---

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

The only runtime dependency is **NumPy**.  Install test dependencies with:

```bash
pip install -e ".[test]"
```

### Run Tests

```bash
pytest tests/ -v
```

All 59 tests pass, including 4 finite-difference gradient checks that verify
the analytic gradients match the numerical approximation.

### Train an MNIST Classifier

```bash
python examples/mnist_demo.py
```

Example output:
```
============================================================
  MiniTorch MNIST Demo
============================================================

Generating synthetic MNIST-like data ...
  Train samples: 1600
  Test  samples: 400
  Input dim:     784
  Classes:       10

Model:
  Sequential(
    Linear(in_features=784, out_features=128, bias=True)
    ReLU()
    Linear(in_features=128, out_features=10, bias=True)
  )
  Total parameters: 101,770

Training for 15 epochs ...

  Epoch  1 | train loss: 0.1471 | train acc: 95.125% | test acc: 99.750%
  ...
  Epoch 15 | train loss: 0.0000 | train acc: 100.000% | test acc: 100.000%

Training finished in 1.4s
Final test accuracy: 100.000%
```

---

## 📦 Project Structure

```
minitorch/
├── minitorch/
│   ├── __init__.py      # Package entry point
│   ├── tensor.py        # Core Tensor + autograd engine
│   ├── nn.py            # Module, Linear, Conv2d, ReLU, Sequential, etc.
│   ├── optim.py         # SGD, Adam
│   └── loss.py          # MSELoss, CrossEntropyLoss
├── tests/
│   ├── test_tensor.py   # 42 tests: ops, gradients, edge cases
│   ├── test_nn.py       # 15 tests: layers, parameters, state dicts
│   └── test_loss.py     # 8 tests: MSE, CrossEntropy
├── examples/
│   └── mnist_demo.py    # End-to-end training demo
├── setup.py
├── requirements.txt
└── README.md
```

---

## 🧪 Testing & Gradient Verification

MiniTorch includes a rigorous test suite (59 tests):

- **Analytic tests** — verify that each operation's forward and backward values
  are mathematically correct.
- **Gradient checks** — compare analytic gradients against central finite
  differences for composite functions involving sums, ReLU, log-softmax,
  and matrix multiplication.
- **Edge cases** — diamond-shaped computation graphs (shared nodes),
  broadcast gradients, and multiple consumers of the same tensor.

---

## 🔬 Why This Matters for Interviews

| Concept Demonstrated         | How MiniTorch Shows It                              |
|------------------------------|-----------------------------------------------------|
| **Computational graphs**      | Dynamic graph built via closure-based backward fns  |
| **Chain rule / backprop**     | Implemented by hand for every op (add, mul, matmul, softmax …) |
| **Topological sort**          | Kahn's algorithm for correct gradient flow order    |
| **Numerical stability**       | Log-Softmax with shift-and-subtract trick           |
| **Software architecture**     | Modular `nn.Module`, parameter discovery, state dicts |
| **Optimization algorithms**   | SGD with momentum, Adam with bias correction        |
| **Testing discipline**        | 59 unit tests, gradient checks, edge case coverage   |

---

## 📄 License

MIT — feel free to use, modify, and share.