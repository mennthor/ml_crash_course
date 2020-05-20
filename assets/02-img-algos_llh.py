# coding: utf-8
"""
Example plots for a gaussian LLH estimation
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy.optimize as sco


# Generate some data points on a straight line with y noise
rng = np.random.RandomState(1337)
x = np.random.normal(1, 1, size=30)


# Perform the likelihood fit, fit mean and stddev for gaussian
def neglogllh(params):
    mean, stddev = params
    return -np.sum(scs.norm.logpdf(x, loc=mean, scale=stddev))


theta0 = [0, 1]  # Seed with standard normal distribution
res = sco.minimize(neglogllh, theta0)
bf_mean, bf_stddev = res.x[0], res.x[1]

# Make grid for pcolormesh loss plot
off = 5
x_grid = np.linspace(bf_mean - off, bf_mean + off, 100)
y_grid = np.linspace(0, bf_stddev + off, 100)
XX, YY = np.meshgrid(x_grid, y_grid)
xx, yy = map(np.ravel, [XX, YY])
ZZ = np.array([neglogllh([xi, yi]) for xi, yi in zip(xx, yy)]).reshape(XX.shape)

# Plot
x_plot = np.linspace(-3, 4, 100)
y_plot = scs.norm.pdf(x_plot, loc=bf_mean, scale=bf_stddev)

fig, (axt, axb) = plt.subplots(2, 1, figsize=(3.5, 6))

# Top: Fitted model and data
axt.vlines(x, 0, 0.1 * np.amax(y_plot), lw=2, ls="-", label="Data")
axt.plot(x_plot, scs.norm.pdf(x_plot, loc=1, scale=1),
         lw=3, c="C7", ls="--", label="True model")
axt.plot(x_plot, y_plot,
         lw=3, c="C3", ls="-", label="Fitted model")
axt.axvline(bf_mean, 0, 1, ls=":", lw=2, c="k")
axt.hist(x, alpha=0.2, color="C7", bins=20, density=True)
axt.set_title("Fitted model")
axt.set_xlabel("x")
axt.set_ylabel("PDF")
axt.legend(loc="upper left")

# Bottom: Loss
img = axb.pcolormesh(XX, YY, np.log(ZZ), cmap="Blues_r")
axb.contour(XX, YY, np.log(ZZ), colors="w",
            levels=[3.9, 4, 4.25, 4.5, 5, 6, 7], linewidths=1, linestyles="--")
axb.plot(bf_mean, bf_stddev,
         ls="", marker="o", mec="k", color="w", label="Minimum")
fig.colorbar(img, ax=axb)
axb.set_title("Loss function")
axb.set_xlabel("mean")
axb.set_ylabel("stddev")
axb.legend(loc="upper center")

fig.tight_layout()
plt.savefig("../build/assets/02-img-algos_llh.png", dpi=200)
# plt.show()
