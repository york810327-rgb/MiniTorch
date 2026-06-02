"""
Optimizers for updating model parameters via gradient descent variants.

Each optimizer implements:
- ``step()``        — apply one update to all parameters.
- ``zero_grad()``   — reset all parameter gradients to None.

Supported optimizers
--------------------
- SGD      — Stochastic Gradient Descent (optionally with momentum & weight decay)
- Adam     — Adaptive Moment Estimation (Kingma & Ba, 2015)

Example
-------
>>> opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
>>> loss = loss_fn(model(x), y)
>>> loss.backward()
>>> opt.step()
>>> opt.zero_grad()
"""

from __future__ import annotations

import numpy as np
from typing import Iterator

from minitorch.nn import Parameter


class SGD:
    r"""Stochastic Gradient Descent with optional momentum and weight decay.

    Update rule (without momentum):
        :math:`\theta_{t+1} = \theta_t - \eta \cdot g_t - \eta \cdot \lambda \cdot \theta_t`

    With momentum:
        :math:`v_{t+1} = \mu \cdot v_t + g_t`
        :math:`\theta_{t+1} = \theta_t - \eta \cdot v_{t+1} - \eta \cdot \lambda \cdot \theta_t`

    where :math:`\eta` is the learning rate, :math:`\mu` is momentum,
    :math:`\lambda` is weight decay, and :math:`g_t` is the gradient.

    Parameters
    ----------
    params : Iterator of Parameter
        Parameters to optimize.
    lr : float, default 0.01
        Learning rate.
    momentum : float, default 0.0
        Momentum factor in [0, 1).
    weight_decay : float, default 0.0
        L2 regularisation coefficient.
    """

    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # velocity buffers (one per param)
        self._v: list[np.ndarray | None] = [None] * len(self.params)

    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.data.grad is None:
                continue
            grad = param.data.grad.copy()

            # Weight decay (L2 regularisation)
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data.data

            # Momentum
            if self.momentum != 0.0:
                if self._v[i] is None:
                    self._v[i] = grad
                else:
                    self._v[i] = self.momentum * self._v[i] + grad  # type: ignore[operator]
                grad = self._v[i]  # type: ignore[assignment]

            # Update
            param.data = type(param.data)(
                param.data.data - self.lr * grad,
                requires_grad=True,
            )

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class Adam:
    r"""Adam (Adaptive Moment Estimation) optimiser.

    Update rule:

        :math:`m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t`
        :math:`v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2`
        :math:`\hat{m}_t = m_t / (1 - \beta_1^t)`
        :math:`\hat{v}_t = v_t / (1 - \beta_2^t)`
        :math:`\theta_t = \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)`

    Parameters
    ----------
    params : Iterator of Parameter
        Parameters to optimize.
    lr : float, default 0.001
        Learning rate.
    betas : tuple of (float, float), default (0.9, 0.999)
        Exponential decay rates for the first / second moment estimates.
    eps : float, default 1e-8
        Small constant for numerical stability.
    weight_decay : float, default 0.0
        L2 regularisation coefficient.
    """

    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._t = 0  # time step counter
        self._m: list[np.ndarray | None] = [None] * len(self.params)
        self._v: list[np.ndarray | None] = [None] * len(self.params)

    def step(self) -> None:
        self._t += 1
        for i, param in enumerate(self.params):
            if param.data.grad is None:
                continue
            grad = param.data.grad.copy()

            # Weight decay
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data.data

            # Update biased first moment estimate
            if self._m[i] is None:
                self._m[i] = (1 - self.beta1) * grad
            else:
                self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad  # type: ignore[operator]

            # Update biased second moment estimate
            if self._v[i] is None:
                self._v[i] = (1 - self.beta2) * (grad * grad)
            else:
                self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad * grad)  # type: ignore[operator]

            # Bias correction
            m_hat = self._m[i] / (1 - self.beta1**self._t)  # type: ignore[operator]
            v_hat = self._v[i] / (1 - self.beta2**self._t)  # type: ignore[operator]

            # Update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # type: ignore[operator]
            param.data = type(param.data)(
                param.data.data - update,
                requires_grad=True,
            )

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()