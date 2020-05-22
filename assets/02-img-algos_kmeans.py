# coding: utf-8
"""
Example plots for a SVM classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


plt.rcParams["lines.linewidth"] = 3
rng = np.random.RandomState(1337)

# Make 2D classification data
xmin, xmax = -3, 3
# Classification truth: Separated by curved region
# Sample in box and set class by comapring to quadratic line
xc = np.sort(rng.uniform(xmin, xmax, size=75))
yc = rng.uniform(xmin, xmax, size=75)
comp = 0.5 * yc**2 - 2
m = xc < comp
classes = np.zeros_like(xc)
classes[m] = 1
# Now smear stuff out a little bit
xc += rng.uniform(-0.5, 0.5, size=len(xc))
yc += rng.uniform(-1., 1., size=len(yc))

# Classify using linear and adaptive (radial basis function) kernel
data = np.vstack([xc, yc]).T
kmeans_2clusters = KMeans(
    n_clusters=2, n_init=1, init=np.array(
        [np.mean(data[m], axis=0), np.mean(data[~m], axis=0)]))
kmeans_3clusters = KMeans(
    n_clusters=3, n_init=1, init=np.array(
        [np.mean(data[m], axis=0), np.mean(data[~m], axis=0), [0, 0]]))
kmeans_2clusters.fit(data)
kmeans_3clusters.fit(data)

# Make boundaries
_x = np.linspace(xmin - 1, xmax + 1, 250)
_y = np.linspace(xmin - 1, xmax + 1, 250)
XX, YY = np.meshgrid(_x, _y)
xx, yy = map(np.ravel, [XX, YY])
grid_points = np.vstack([xx, yy]).T
preds_2classes = kmeans_2clusters.predict(grid_points)
preds_3classes = kmeans_3clusters.predict(grid_points)

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(3.5, 6))

colors = ["C1", "C0", "C2"]

# Top: 2 classes
axt.scatter(xc[m], yc[m], c=colors[0], s=20)
axt.scatter(xc[~m], yc[~m], c=colors[1], s=20)
# Contours
axt.contourf(XX, YY, preds_2classes.reshape(XX.shape),
             levels=[0., 0.5, 1.], vmin=0, vmax=1, alpha=0.1, colors=colors[:2])
axt.contour(XX, YY, preds_2classes.reshape(XX.shape),
            levels=[0.5], linewidths=2, colors=["C7"])
# Centers
axt.plot(
    kmeans_2clusters.cluster_centers_[:, 0],
    kmeans_2clusters.cluster_centers_[:, 1], ls="",
    marker="o", mew=1, mec="k", ms=5, c="w", label="Centers")

axt.set_title("2 Clusters")
axt.set_xlabel("Feature 1")
axt.set_ylabel("Feature 2")
axt.legend(prop={"size": 6})

# Bottom: 3 classes
axb.scatter(xc[m], yc[m], c=colors[0], s=20)
axb.scatter(xc[~m], yc[~m], c=colors[1], s=20)
# Contours, shuffle colors to match truth colors
colors = np.array(colors)[[1, 2, 0]]
axb.contourf(
    XX, YY, preds_3classes.reshape(XX.shape),
    levels=[0., 0.5, 1.5, 2.], vmin=0, vmax=2, alpha=0.1, colors=colors)
axb.contour(XX, YY, preds_3classes.reshape(XX.shape),
            levels=[0.5, 1.5], linewidths=2, colors=["C7", "C7"])
# Centers
axb.plot(
    kmeans_3clusters.cluster_centers_[:, 0],
    kmeans_3clusters.cluster_centers_[:, 1], ls="",
    marker="o", mew=1, mec="k", ms=5, c="w", label="Centers")

axb.set_title("3 Clusters")
axb.set_xlabel("Feature 1")
axb.set_ylabel("Feature 2")
axb.legend(prop={"size": 6})

fig.tight_layout()
plt.savefig("../build/assets/02-img-algos_kmeans.png", dpi=200)
# plt.show()
