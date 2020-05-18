# coding: utf-8
"""
Example plots for binary cross entropy loss and mean squared error loss.
https://www.deeplearningdemystified.com/article/fdl-3
"""
import numpy as np
import matplotlib.pyplot as plt


def binary_cross_entropy_loss(p, p_true):
    """ p in [0, 1], p_true in {0, 1} """
    return -p_true * np.log(p) - (1. - p_true) * np.log(1. - p)


def mse_loss_func(y, y_true):
    """ Loss for single example point """
    return 0.5 * (y_true - y)**2


def huber_loss_func(y, y_true, delta):
    """ https://en.wikipedia.org/wiki/Huber_loss """
    abs_y = abs_loss_func(y, y_true)
    m = abs_y < delta
    out = np.empty_like(y)
    out[m] = mse_loss_func(y[m], y_true)
    out[~m] = delta * abs_y[~m] - 0.5 * delta**2
    return out


def abs_loss_func(y, y_true):
    """ https://en.wikipedia.org/wiki/Huber_loss """
    return np.abs(y_true - y)


# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(4, 6))

# Top plot: Categorial loss
p = np.linspace(1e-5, 1 - 1e-5, 200)
be_loss_class0 = binary_cross_entropy_loss(p, 0)
be_loss_class1 = binary_cross_entropy_loss(p, 1)

axt.set_title("Binary CE loss = "
              + "- $y_i \\ln(\\hat{{y}}_i) - (1 - y_i) \\ln(1 - \\hat{{y}}_i)$")
axt.plot(p, be_loss_class0, ls="-", c="C0", label="True class 0", lw=2.5)
axt.plot(p, be_loss_class1, ls="-", c="C1", label="True class 1", lw=2.5)

axt.set_xlabel("Predicted probability $p$ belonging to class 1")
axt.set_ylabel("Loss")
axt.grid()
axt.legend()

# Bottom plot: Logit function
delta = 1
y = np.linspace(-3, 3, 200)
mse_loss = mse_loss_func(y, y_true=0)
huber_loss = huber_loss_func(y, y_true=0, delta=delta)
abs_loss = abs_loss_func(y, y_true=0)

axb.set_title("Various regression losses")
axb.plot(y, mse_loss, ls="-", c="C0", label="MSE loss", lw=2.5)
axb.plot(y, huber_loss, ls="-", c="C1", label="Huber loss", lw=2.5)
axb.plot(y, abs_loss, ls="-", c="C2", label="Absolute loss", lw=2.5)
axb.axvline(-delta, 0, 1, ls="--", lw=1, c="k")
axb.axvline(delta, 0, 1, ls="--", lw=1, c="k")

axb.set_xlabel("Predicted error $y - \\hat{{y}}$")
axb.set_ylabel("Loss")
axb.grid()
axb.legend()

fig.tight_layout()
plt.savefig("../build/assets/02-img-loss_functions.png", dpi=200)
# plt.show()
