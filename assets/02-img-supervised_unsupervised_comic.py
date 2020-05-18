# coding: utf-8
"""
Comic plot supervised and unsupervised lesrners.
Idea from: https://www.researchgate.net/figure/Examples-of-Supervised-Learning-Linear-Regression-and-Unsupervised-Learning_fig3_336642133
"""
import numpy as np
import matplotlib.pyplot as plt


# Generate some blob data with two classes
rng = np.random.RandomState(1337)
c0x, c0y = 0, 0
c1x, c1y = 2, 2
stddev = 0.5
x0, y0 = rng.normal(c0x, stddev, size=25), rng.normal(c0y, stddev, size=25)
x1, y1 = rng.normal(c1x, stddev, size=25), rng.normal(c1y, stddev, size=25)

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(4, 6))

# Top plot: Supervised
axt.set_title("Supervised")
axt.scatter(x0, y0, marker="o", c="C0", label="Class 0")
axt.scatter(x1, y1, marker="o", c="C1", label="Class 1")
axt.plot([-1, 3], [3, -1], ls="--", lw=2, c="C7")

axt.set_xlabel("$x$")
axt.set_ylabel("$y$")
axt.grid()
axt.legend(loc="upper left")

# Bottom plot: Unsupervised
axb.set_title("Unsupervised")
axb.scatter(x0, y0, marker="o", c="C7")
axb.scatter(x1, y1, marker="o", c="C7")
circ0 = plt.Circle((c0x, c0y), 1.5 * stddev,
                   edgecolor="C7", lw=2, linestyle="--", facecolor="none")
circ1 = plt.Circle((c1x, c1y), 1.5 * stddev,
                   edgecolor="C7", lw=2, linestyle="--", facecolor="none")
axb.add_artist(circ0)
axb.add_artist(circ1)

axb.set_xlabel("$x$")
axb.set_ylabel("$y$")
axb.grid()

fig.tight_layout()
plt.savefig("../build/assets/02-img-supervised_unsupervised_comic.png", dpi=200)
# plt.show()
