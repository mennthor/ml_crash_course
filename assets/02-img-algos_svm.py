# coding: utf-8
"""
Example plots for a SVM classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


plt.rcParams["lines.linewidth"] = 3
rng = np.random.RandomState(1337)

# Make 2D classification data
xmin, xmax = -3, 3
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

# Classify using linear and adaptive (radial basis function) kernel
data = np.vstack([xc, yc]).T
svm_nonlin = SVC(kernel="rbf")
svm_lin = SVC(kernel="linear")
svm_nonlin.fit(data, classes)
svm_lin.fit(data, classes)

# Make boundaries
_x = np.linspace(xmin - 1, xmax + 1, 250)
_y = np.linspace(xmin - 1, xmax + 1, 250)
XX, YY = np.meshgrid(_x, _y)
xx, yy = map(np.ravel, [XX, YY])
grid_points = np.vstack([xx, yy]).T
# preds_nonlin = svm_nonlin.predict_proba(grid_points)[:, 1]
preds_nonlin_classes = svm_nonlin.predict(grid_points)
# preds_lin = svm_lin.predict_proba(grid_points)[:, 1]
preds_lin_classes = svm_lin.predict(grid_points)

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(3.5, 6))

colors = ["C1", "C0"]

# Top: RBF kernel
axt.scatter(xc[m], yc[m], c=colors[1])
axt.scatter(xc[~m], yc[~m], c=colors[0])
# Contours
axt.contourf(XX, YY, preds_nonlin_classes.reshape(XX.shape),
             levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axt.contour(XX, YY, preds_nonlin_classes.reshape(XX.shape),
            levels=[0.5], linewidths=2, colors=["C7"])

axt.set_title("Non-linear kernel (RBF)")
axt.set_xlabel("Feature 1")
axt.set_ylabel("Feature 2")

# Bottom: linear kernel
axb.scatter(xc[m], yc[m], c=colors[1])
axb.scatter(xc[~m], yc[~m], c=colors[0])
# Contours
axb.contourf(XX, YY, preds_lin_classes.reshape(XX.shape),
             levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axb.contour(XX, YY, preds_lin_classes.reshape(XX.shape),
            levels=[0.5], linewidths=2, colors=["C7"])

# Get the separator line and the margins from the coefficients. Copied 1:1 from:
#  https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
# This works only for linear kernels
w = svm_lin.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xmin, xmax)
yy = a * xx - (svm_lin.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors (margin away from hyperplane in direction
# perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
# 2-d.
margin = 1 / np.sqrt(np.sum(svm_lin.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

# plot the line, the points, and the nearest vectors to the plane
axb.plot(xx, yy, c="C7", ls="-", lw=2)
axb.plot(xx, yy_down, c="C7", ls="--", lw=2)
axb.plot(xx, yy_up, c="C7", ls="--", lw=2)
axb.set_ylim(xmin, xmax)

axb.set_title("Linear kernel")
axb.set_xlabel("Feature 1")
axb.set_ylabel("Feature 2")

fig.tight_layout()
plt.savefig("../build/assets/02-img-algos_svm.png", dpi=200)
# plt.show()
