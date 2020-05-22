# coding: utf-8
"""
Example plots for a naive bayes classification using a binary class Bernoulli
model (same as logistic regression loss)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

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
dec_tree = DecisionTreeRegressor(max_depth=max_depth, random_state=rng)
dec_tree_overfit = DecisionTreeRegressor(max_depth=None, random_state=rng)

# Regress predicted yvalues
_x = np.linspace(xmin - 0.1, xmax + 0.1, 500)

dec_tree.fit(x.reshape(len(x), 1), y_quad)
dec_tree_overfit.fit(x.reshape(len(x), 1), y_quad)
_y_quad = dec_tree.predict(_x.reshape(len(_x), 1))
_y_quad_over = dec_tree_overfit.predict(_x.reshape(len(_x), 1))

dec_tree.fit(x.reshape(len(x), 1), y_sine)
dec_tree_overfit.fit(x.reshape(len(x), 1), y_sine)
_y_sine = dec_tree.predict(_x.reshape(len(_x), 1))
_y_sine_over = dec_tree_overfit.predict(_x.reshape(len(_x), 1))

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(3.5, 6))

# Top: Quadratic data regressor
axt.plot(_x, quadratic(_x), lw=2, ls=":", c="C7", label="Truth")
axt.plot(_x, _y_quad, ls="-", c="C0", label="max depth = {}".format(max_depth))
axt.plot(_x, _y_quad_over, ls="--", c="C1", label="Full depth")
axt.plot(x, y_quad, marker="o", ls="", c="C7", mec="k", mew=1, label="Data")

axt.set_title("Quadratic truth")
axt.set_xlabel("$x$")
axt.set_ylabel("$y$")
axt.legend(loc="upper center", prop={'size': 6}, ncol=2)

# Bottom: Sine data regressor
axb.plot(_x, sine(_x), lw=2, ls=":", c="C7", label="Truth")
axb.plot(_x, _y_sine, ls="-", c="C0", label="max depth = {}".format(max_depth))
axb.plot(_x, _y_sine_over, ls="--", c="C1", label="Full depth")
axb.plot(x, y_sine, marker="o", ls="", c="C7", mec="k", mew=1, label="Data")

axb.set_title("Sine truth")
axb.set_xlabel("$x$")
axb.set_ylabel("$y$")
axb.legend(loc="lower right", prop={'size': 6}, ncol=1)

fig.tight_layout()
plt.savefig("../build/assets/02-img-algos_tree_regression.png", dpi=200)
# plt.show()
