# coding: utf-8
"""
Example plots for a kNN classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


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

# Classify
data = np.vstack([xc, yc]).T
knn = KNeighborsClassifier(n_neighbors=5)
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn.fit(data, classes)
knn_1.fit(data, classes)

# Make boundaries
_x = np.linspace(xmin - 1, xmax + 1, 250)
_y = np.linspace(xmin - 1, xmax + 1, 250)
XX, YY = np.meshgrid(_x, _y)
xx, yy = map(np.ravel, [XX, YY])
grid_points = np.vstack([xx, yy]).T
preds = knn.predict_proba(grid_points)[:, 1]
preds_classes = knn.predict(grid_points)
preds__1 = knn_1.predict_proba(grid_points)[:, 1]
preds_classes_1 = knn_1.predict(grid_points)

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(3.5, 6))

colors = ["C1", "C0"]

# Top k=3
axt.scatter(xc[m], yc[m], c=colors[1])
axt.scatter(xc[~m], yc[~m], c=colors[0])
# Contours
axt.contourf(XX, YY, preds_classes.reshape(XX.shape),
             levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axt.contour(XX, YY, preds.reshape(XX.shape),
            levels=[0.5], linewidths=2, colors=["C7"])

axt.set_title("$k=5$")
axt.set_xlabel("Feature 1")
axt.set_ylabel("Feature 2")

# Bottom k=1
axb.scatter(xc[m], yc[m], c=colors[1])
axb.scatter(xc[~m], yc[~m], c=colors[0])
# Contours
axb.contourf(XX, YY, preds_classes_1.reshape(XX.shape),
             levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors)
axb.contour(XX, YY, preds_classes_1.reshape(XX.shape),
            levels=[0.5], linewidths=2, colors=["C7"])

axb.set_title("$k=1$")
axb.set_xlabel("Feature 1")
axb.set_ylabel("Feature 2")

fig.tight_layout()
plt.savefig("../build/assets/02-img-algos_knn.png", dpi=200)
# plt.show()
