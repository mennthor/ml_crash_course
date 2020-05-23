# coding: utf-8
"""
Example plots for a naive bayes classification using a binary class Bernoulli
model (same as logistic regression loss)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

plt.rcParams["lines.linewidth"] = 3
rng = np.random.RandomState(1337)


# Regression truth: Quadratic and sine
def quadratic(x):
    return x**2 + 1


def sine(x):
    return 2 * np.sin(x) + 1


xmin, xmax = -3, 3
x = np.sort(rng.uniform(xmin, xmax, size=20))
y_quad = quadratic(x) + rng.uniform(-2, 2, size=len(x))
y_sine = sine(x) + rng.uniform(-0.5, 0.5, size=len(x))

# Use fit a regularized and overfitting tree to both datasets
max_depth = 2
nn = MLPRegressor(alpha=0.1, hidden_layer_sizes=(5, 5), max_iter=2000,
                  random_state=rng, solver="lbfgs")
nn_overfit = MLPRegressor(alpha=0., hidden_layer_sizes=(100, 100, 100, 100),
                          max_iter=2000, solver="lbfgs", random_state=rng)

# Regress predicted yvalues
_x = np.linspace(xmin - 0.1, xmax + 0.1, 500)

nn.fit(x.reshape(len(x), 1), y_quad)
nn_overfit.fit(x.reshape(len(x), 1), y_quad)
_y_quad = nn.predict(_x.reshape(len(_x), 1))
_y_quad_over = nn_overfit.predict(_x.reshape(len(_x), 1))

nn.fit(x.reshape(len(x), 1), y_sine)
nn_overfit.fit(x.reshape(len(x), 1), y_sine)
_y_sine = nn.predict(_x.reshape(len(_x), 1))
_y_sine_over = nn_overfit.predict(_x.reshape(len(_x), 1))

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(3.5, 6))

# Top: Quadratic data regressor
axt.plot(_x, quadratic(_x), lw=2, ls=":", c="C7", label="Truth")
axt.plot(_x, _y_quad,
         ls="-", c="C0", label="Small, regularized".format(max_depth))
axt.plot(_x, _y_quad_over, ls="--", c="C1", label="Large, unregularized")
axt.plot(x, y_quad, marker="o", ls="", c="C7", mec="k", mew=1, label="Data")

axt.set_title("Quadratic truth")
axt.set_xlabel("$x$")
axt.set_ylabel("$y$")
axt.set_ylim(-1, 6)
axt.legend(loc="upper center", prop={'size': 6}, ncol=2)

# Bottom: Sine data regressor
axb.plot(_x, sine(_x), lw=2, ls=":", c="C7", label="Truth")
axb.plot(_x, _y_sine,
         ls="-", c="C0", label="Small, regularized".format(max_depth))
axb.plot(_x, _y_sine_over, ls="--", c="C1", label="Large, unregularized")
axb.plot(x, y_sine, marker="o", ls="", c="C7", mec="k", mew=1, label="Data")

axb.set_title("Sine truth")
axb.set_xlabel("$x$")
axb.set_ylabel("$y$")
axb.set_ylim(-2, 4)
axb.legend(loc="lower right", prop={'size': 6}, ncol=1)

fig.tight_layout()
plt.savefig("../build/assets/02-img-algos_nn_regression.png", dpi=200)
# plt.show()
