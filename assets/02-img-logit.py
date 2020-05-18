# coding: utf-8
"""
Example plots for a logit function and how you can't simply fit a straight line
to categorial data
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs


def sigmoid_func(x, mu, sigma):
    return 1. / (1. + np.exp(-(x - mu) / sigma))


# Generate some data points from a logisitic model
rng = np.random.RandomState(1337)
# Draw categorial data from logistic (sigmoid) function
mu = 1
sigma = 0.25
x = rng.uniform(-1, 3, size=50)
probs = sigmoid_func(x, mu, sigma)
u = rng.uniform(size=len(x))
m = probs > u
y = np.zeros_like(x, dtype=int)
y[m] = 1


# Perform the line fit to categorial data
slope, intercept, r_value, p_value, std_err = scs.linregress(x, y)

# Plot
fig, (axt, axb) = plt.subplots(2, 1, figsize=(4, 6))

# Top plot: Fit line to categorial data
x_plot = np.linspace(-1, 4, 200)
y_plot = sigmoid_func(x_plot, mu, sigma)
y_plot_est = slope * x_plot + intercept

axt.set_title("Using linear model for categorial data")
axt.plot(x[m], y[m], ls="", marker="o", c="C1", label="Class 1")
axt.plot(x[~m], y[~m], ls="", marker="o", c="C0", label="Class 0")
axt.plot(x_plot, y_plot, c="C7", lw=3, ls="--", label="True model")
axt.plot(x_plot, y_plot_est, c="C3", lw=3, label="Fitted line")

axt.set_xlabel("x")
axt.set_ylabel("class probability")
axt.legend()

# Bottom plot: Logit function
p = np.linspace(1e-5, 1. - 1e-5, 200)
logit = np.log(p / (1. - p))

axb.set_title("Logit function $\\ln(p / (1 - p))$")
axb.plot(p, logit)
axb.set_xlabel("$p$")
axb.set_ylabel("Logit($p$)")
axb.grid()

fig.tight_layout()
plt.savefig("../build/assets/02-img-logit.png", dpi=200)
# plt.show()
