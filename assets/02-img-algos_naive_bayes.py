# coding: utf-8
"""
Example plots for a naive bayes classification using a binary class Bernoulli
model (same as logistic regression loss)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.naive_bayes import GaussianNB


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

# Make a nicely seperable dataset
data2 = scs.multivariate_normal.rvs(
    [-1, 0], [[1, 0.9], [0.9, 1]], random_state=rng, size=25)
classes2 = np.zeros(len(data2))
data2 = np.concatenate((data2, scs.multivariate_normal.rvs(
    [1, 0], [[1, 0.9], [0.9, 1]], random_state=rng, size=25)), axis=0)
classes2 = np.concatenate((classes2, np.ones(len(classes2))))

# Classify
data = np.vstack([xc, yc]).T
nbayes = GaussianNB()
nbayes_lin = GaussianNB()
nbayes.fit(data, classes)
nbayes_lin.fit(data2, classes2)

# Make boundaries
_x = np.linspace(xmin - 1, xmax + 1, 250)
_y = np.linspace(xmin - 1, xmax + 1, 250)
XX, YY = np.meshgrid(_x, _y)
xx, yy = map(np.ravel, [XX, YY])
grid_points = np.vstack([xx, yy]).T
preds = nbayes.predict_proba(grid_points)[:, 1]
preds_classes = nbayes.predict(grid_points)
preds_lin = nbayes_lin.predict_proba(grid_points)[:, 1]
preds_classes_lin = nbayes_lin.predict(grid_points)

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(3.5, 6))

colors = ["C1", "C0"]

# Top: not separable
axt.scatter(xc[m], yc[m], c=colors[1])
axt.scatter(xc[~m], yc[~m], c=colors[0])
# Contours
axt.contourf(XX, YY, preds_classes.reshape(XX.shape),
             levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axt.contour(XX, YY, preds.reshape(XX.shape),
            levels=[0.5], linewidths=2, colors=["C7"])

axt.set_title("Not linearly separable")
axt.set_xlabel("Feature 1")
axt.set_ylabel("Feature 2")

# Bottom: separable
m = (classes2 == 1)
axb.scatter(data2[:, 0][m], data2[:, 1][m], c=colors[1])
axb.scatter(data2[:, 0][~m], data2[:, 1][~m], c=colors[0])
# Contours
axb.contourf(XX, YY, preds_classes_lin.reshape(XX.shape),
             levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axb.contour(XX, YY, preds_classes_lin.reshape(XX.shape),
            levels=[0.5], linewidths=2, colors=["C7"])

axb.set_title("Linearly separable")
axb.set_xlabel("Feature 1")
axb.set_ylabel("Feature 2")

fig.tight_layout()
plt.savefig("../build/assets/02-img-algos_naive_bayes.png", dpi=200)
# plt.show()
