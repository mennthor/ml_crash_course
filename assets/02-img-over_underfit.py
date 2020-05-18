#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

plt.rcParams["lines.linewidth"] = 3

rng = np.random.RandomState(1337)
# Trying to reproduce something similar to
#   https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks#classification-metrics


# Make 1D regression and 2D classification data.

# Regression truth: Quadratic
def reg_truth(x):
    return x**2 + 1


xmin, xmax = -3, 3
x = np.sort(rng.uniform(xmin, xmax, size=20))
y = reg_truth(x) + rng.uniform(-2, 2, size=len(x))

# Classification truth: Separated by curved region
# Sample in box and set class by comapring to quadratic line
xc = np.sort(rng.uniform(xmin, xmax, size=75))
yc = rng.uniform(xmin, xmax, size=75)
comp = 0.5 * yc**2 - 2
m = xc > comp
classes = np.zeros_like(xc)
classes[m] = 1
# Now smear stuff out a little bit
xc += rng.uniform(-0.5, 0.5, size=len(xc))
yc += rng.uniform(-1., 1., size=len(yc))


# Regress with simple line (underfit), quadratic (right) and 11th order
# polynomial (overfit)
reg_coeff_lin = np.polyfit(x, y, deg=1)
reg_coeff_quad = np.polyfit(x, y, deg=2)
reg_coeff_11th = np.polyfit(x, y, deg=11)


# Classification
# Underfit with logistic regression
data = np.vstack([xc, yc]).T
logreg = LogisticRegression(random_state=rng)
logreg.fit(data, classes)

# Fake a "just right" using the true boundary
net_right = MLPClassifier(hidden_layer_sizes=(5, 10, 5), alpha=1,
                          random_state=rng, max_iter=500, solver="adam")
net_right.fit(data, classes)

# Use an unregularized, way too big net to see overfitting
net_over = MLPClassifier(hidden_layer_sizes=(200, 200), alpha=0,
                         random_state=rng, max_iter=500, solver="lbfgs")
net_over.fit(data, classes)

# Make boundaries
_x = np.linspace(xmin - 1, xmax + 1, 500)
_y = np.linspace(xmin - 1, xmax + 1, 500)
XX, YY = np.meshgrid(_x, _y)
xx, yy = map(np.ravel, [XX, YY])
grid_points = np.vstack([xx, yy]).T

# Logistic regression
preds_logreg = logreg.predict(grid_points)
preds_logreg_classes = np.round(preds_logreg)

# Just right
preds_net_right = net_right.predict(grid_points)
preds_net_right_classes = np.round(preds_net_right)

# Network overfit
preds_net_over = net_over.predict(grid_points)
preds_net_over_classes = np.round(preds_net_over)


_x = np.linspace(xmin, xmax, 200)
_y_lin = np.polyval(reg_coeff_lin, _x)
_y_quad = np.polyval(reg_coeff_quad, _x)
_y_11th = np.polyval(reg_coeff_11th, _x)

fig, axes = plt.subplots(2, 3, figsize=(9, 4.5), sharex=True)
axtl, axtc, axtr = axes[0]
axbl, axbc, axbr = axes[1]

# Top row: Regression plots
for axi in axes[0]:
    axi.plot(x, y, ls="", marker="o", c="C2")
axtl.plot(_x, _y_lin, ls="-", c="C7")
axtc.plot(_x, _y_quad, ls="-", c="C7")
axtr.plot(_x, _y_11th, ls="-", c="C7")
axtl.set_title("Underfit")
axtc.set_title("Just right")
axtr.set_title("Overfit")

for i, axi in enumerate(axes[0]):
    axi.set_ylim(0, 10)
    if i > 0:
        axi.set_yticklabels("")

# Bottom row: Classification plots
for axi in axes[1]:
    axi.scatter(xc[m], yc[m], c="C0")
    axi.scatter(xc[~m], yc[~m], c="C1")
# Decision contours
colors = ["C1", "C0"]
# Logisitic regression
axbl.contourf(XX, YY, preds_logreg_classes.reshape(XX.shape),
              levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axbl.contour(XX, YY, preds_logreg.reshape(XX.shape),
             levels=[0.5], linewidths=2, colors=["C7"])
# Just right
axbc.contourf(XX, YY, preds_net_right_classes.reshape(XX.shape),
              levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axbc.contour(XX, YY, preds_net_right.reshape(XX.shape),
             levels=[0.5], linewidths=2, colors=["C7"])
# Network overfit
axbr.contourf(XX, YY, preds_net_over_classes.reshape(XX.shape),
              levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axbr.contour(XX, YY, preds_net_over.reshape(XX.shape),
             levels=[0.5], linewidths=2, colors=["C7"])

for i, axi in enumerate(axes[1]):
    axi.set_xlabel("x")
    axi.set_ylim(-3, 3)
    if i > 0:
        axi.set_yticklabels("")

# Common setup
axtl.set_ylabel("y")
axbl.set_ylabel("y")
axtr.yaxis.set_label_position("right")
axtr.yaxis.tick_right()
axbr.yaxis.set_label_position("right")
axbr.yaxis.tick_right()
axtr.set_ylabel("Regression")
axbr.set_ylabel("Classification")

fig.tight_layout()
plt.savefig("../build/assets/02-img-over_underfit.png", dpi=200)
# plt.show()
