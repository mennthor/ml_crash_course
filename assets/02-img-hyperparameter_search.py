#!/usr/bin/env python
# coding: utf-8

"""
Hyperparameter tuning random vs grid search.
Trying to do something like
    https://blogs.sas.com/content/subconsciousmusings/2016/09/20/local-search-optimization-for-hyperparameter-tuning/#prettyPhoto
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci

plt.rcParams["lines.linewidth"] = 3
rng = np.random.RandomState(1337)
# Make 1D regression and 2D classification data.


# Make two "distributions" interpolating with splines
xmin, xmax = 0, 1
x = np.linspace(xmin, xmax, 10)
y1 = np.array([0.05, 0.1, 0.3, 0.5, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05]) + 0.01
y2 = np.array([0.1, 0.2, 0.3, 0.5, 0.2, 0.1, 0.1, 0.2, 0.1, 0.05]) + 0.01

spl1 = sci.InterpolatedUnivariateSpline(x, y1, bbox=[xmin, xmax], ext="zeros")
spl2 = sci.InterpolatedUnivariateSpline(x, y2, bbox=[xmin, xmax], ext="zeros")

x_grid = np.linspace(0.1 * xmax, 0.9 * xmax, 3)
y_grid = np.linspace(0.1 * xmax, 0.9 * xmax, 3)
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

x_rnd = rng.uniform(0.1 * xmax, 0.9 * xmax, len(x_grid)**2)
y_rnd = rng.uniform(0.1 * xmax, 0.9 * xmax, len(x_grid)**2)


# Plot
fig = plt.figure(figsize=(3, 6))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(5, 2, width_ratios=(7, 2), height_ratios=(2, 7, 2, 2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

ax0 = fig.add_subplot(gs[1, 0])
ax0t = fig.add_subplot(gs[0, 0])
ax0r = fig.add_subplot(gs[1, 1])

_x = np.linspace(xmin, xmax, 100)
_y1 = spl1(_x)
_y2 = spl2(_x)

# Main plot top, show sampling grid
ax0.plot(xx_grid.flatten(), yy_grid.flatten(),
         marker="o", c="C7", mec="k", ms=10, ls="")
for xi in x_grid:
    ax0.axvline(xi, 0, 1, ls="--", lw=1, c="C7")
    ax0.axhline(xi, 0, 1, ls="--", lw=1, c="C7")

# Top and left: Show distributions with grid points
# Top
ax0t.fill_between(_x, _y1, color="C0")
for xi in x_grid:
    ax0t.plot(xi, 0.1, marker="o", c="w", ls="", mec="C7", ms=10)
# Bottom
ax0r.fill_between(_y2, _x, color="C1")
for xi in x_grid:
    ax0r.plot(0.1, xi, marker="o", c="w", ls="", mec="C7", ms=10)


for axi in [ax0, ax0t, ax0r]:
    axi.set_xlim(xmin, xmax)
    axi.set_ylim(xmin, xmax)
ax0r.set_ylim(xmin + 0.01, xmax)  # Hotfix for weird fill artifact
ax0.tick_params(axis="x", labelbottom=False)
ax0.tick_params(axis="y", labelleft=False)
ax0.set_xlabel("Feature 1", labelpad=12)
ax0.set_ylabel("Feature 2", labelpad=10)

ax0t.axis("off")
ax0r.axis("off")


# Main plot bottom, show sampling grid
ax1 = fig.add_subplot(gs[4, 0])
ax1t = fig.add_subplot(gs[3, 0])
ax1r = fig.add_subplot(gs[4, 1])

ax1.plot(x_rnd, y_rnd, marker="o", c="C7", mec="k", ms=10, ls="")
for xi, yi in zip(x_rnd, y_rnd):
    ax1.axvline(xi, 0, 1, ls="--", lw=1, c="C7")
    ax1.axhline(yi, 0, 1, ls="--", lw=1, c="C7")

# Top and left: Show distributions with grid points
# Top
ax1t.fill_between(_x, _y1, color="C0")
for xi in x_rnd:
    ax1t.plot(xi, 0.1, marker="o", c="w", ls="", mec="C7", ms=10)
# Bottom
ax1r.fill_between(_y2, _x, color="C1")
for yi in y_rnd:
    ax1r.plot(0.1, yi, marker="o", c="w", ls="", mec="C7", ms=10)


for axi in [ax1, ax1t, ax1r]:
    axi.set_xlim(xmin, xmax)
    axi.set_ylim(xmin, xmax)
ax1r.set_ylim(xmin + 0.01, xmax)  # Hotfix for weird fill artifact
ax1.tick_params(axis="x", labelbottom=False)
ax1.tick_params(axis="y", labelleft=False)
ax1.set_xlabel("Feature 1", labelpad=12)
ax1.set_ylabel("Feature 2", labelpad=10)
ax1t.set_title("Random Search", y=0.55)
ax0t.set_title("Grid Search", y=0.55)

ax1t.axis("off")
ax1r.axis("off")

plt.savefig("../build/assets/02-img-hyperparameter_search.png", dpi=200)
# plt.show()
