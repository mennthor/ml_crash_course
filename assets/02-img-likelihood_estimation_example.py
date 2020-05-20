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

# Plot
x_plot = np.linspace(-3, 4, 100)
y_plot = scs.norm.pdf(x_plot, loc=bf_mean, scale=bf_stddev)
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
ax.set_title("Fitted mean: {:.2f}, stddev: {:.2f} ".format(bf_mean, bf_stddev)
             + "(true mean: {:.0f}, stddev: {:.0f})".format(1, 1))
ax.vlines(x, 0, 0.1 * np.amax(y_plot), lw=2, ls="-", label="Data points")
ax.plot(x_plot, scs.norm.pdf(x_plot, loc=1, scale=1),
        lw=3, c="C7", ls="--", label="True model")
ax.plot(x_plot, y_plot,
        lw=3, c="C3", ls="-", label="Fitted model")
ax.axvline(bf_mean, 0, 1, ls=":", lw=2, c="k")
ax.hist(x, alpha=0.2, color="C7", bins=20, density=True)
ax.set_xlabel("x")
ax.set_ylabel("PDF")


ax.legend()
plt.savefig("../build/assets/02-img-likelihood_estimation_example.png", dpi=200)
# plt.show()
