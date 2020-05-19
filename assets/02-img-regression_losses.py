#!/usr/bin/env python
# coding: utf-8

"""
Example plots for using MSE MAE and Huber loss in same regression task.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco


plt.rcParams["lines.linewidth"] = 3
rng = np.random.RandomState(1337)
# Make 1D regression and 2D classification data.


def truth_func(x, a=0, b=0):
    """ Linear function """
    return a * x + b


def mse_loss_func(params, x, y):
    """ Loss for single example point """
    a, b = params
    return np.mean(0.5 * (y - truth_func(x, a, b))**2)


def abs_loss_func(params, x, y):
    """ https://en.wikipedia.org/wiki/Huber_loss """
    a, b = params
    return np.mean(np.abs(y - truth_func(x, a, b)))


def huber_loss_func(params, x, y, delta):
    """ https://en.wikipedia.org/wiki/Huber_loss """
    a, b = params
    abs_y = np.abs(y - truth_func(x, a, b))
    m = abs_y < delta
    out = np.zeros_like(y)
    out[m] = 0.5 * (y[m] - truth_func(x[m], a, b))**2
    out[~m] = delta * abs_y[~m] - 0.5 * delta**2
    return np.sum(out)


# Prepare pseudo data
true_a, true_b = 0.5, 0.25
xmin, xmax = -3, 3
x = np.sort(rng.uniform(xmin, xmax, size=20))
y = truth_func(x, true_a, true_b) + rng.uniform(-0.5, 0.5, size=len(x))
# Add leveraging outliers manually to demonstrate effects of different losses
outlier_idx = [1, 18]
y[outlier_idx[0]] = truth_func(x[outlier_idx[0]], true_a, true_b) + 4.
y[outlier_idx[1]] = truth_func(x[outlier_idx[1]], true_a, true_b) - 4.

# Do the regression
x0 = np.array([2., -1.])  # Detune start point from truth
res_mse = sco.minimize(mse_loss_func, x0, args=(x, y,))
res_abs = sco.minimize(abs_loss_func, x0, args=(x, y,))
huber_delta = 0.5
res_huber = sco.minimize(huber_loss_func, x0, args=(x, y, huber_delta))

# Plot
_x = np.linspace(xmin, xmax, 2)
_y_true = truth_func(_x, true_a, true_b)
_y_mse = np.polyval(res_mse.x, _x)
_y_abs = np.polyval(res_abs.x, _x)
_y_huber = np.polyval(res_huber.x, _x)

fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

ax.plot(_x, _y_true, ls="--", c="C7", label="Truth")
ax.plot(x, y, ls="", marker="o", ms=10, mec="k", mew=1, c="C7", label="Data")
ax.plot(x[outlier_idx], y[outlier_idx], ls="", marker="o", ms=10, mec="k",
        mew=1, c="C3", label="Outlier")
ax.plot(_x, _y_mse, ls="-", c="C0", label="MSE loss")
ax.plot(_x, _y_abs, ls="-", c="C1", label="MAE loss")
ax.plot(_x, _y_huber, ls="-", c="C2",
        label="Huber loss ($\\delta={:.1f}$)".format(huber_delta))

ax.set_xlim(xmin, xmax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(ncol=2)
ax.grid()

fig.tight_layout()
plt.savefig("../build/assets/02-img-regression_losses.png", dpi=200)
# plt.show()
