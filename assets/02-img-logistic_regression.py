# coding: utf-8
"""
Example plots for a logit function and how you can't simply fit a straight line
to categorial data
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco


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


# Perform the logistic regression manually using the LLH
def neglogllh(params):
    mu, sigma = params
    p = sigmoid_func(x, mu, sigma)
    # Just for numerical sanity
    p[p < 1e-5] = 1e-5
    p[p > 1 - 1e-5] = 1 - 1e-5
    return -np.sum(y * np.log(p) + (1. - y) * np.log(1. - p))


theta0 = [2, 0.5]
res = sco.minimize(neglogllh, theta0)
bf_mu, bf_sigma = res.x[0], res.x[1]

# Plot
x_plot = np.linspace(-1, 3, 200)
y_plot = sigmoid_func(x_plot, mu, sigma)
y_plot_est = sigmoid_func(x_plot, bf_mu, bf_sigma)
decision_boundary = x_plot[np.where(y_plot_est > 0.5)[0][0]]

fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

plt.title("Fitted loc: {:.2f}, scale: {:.2f} ".format(bf_mu, bf_sigma)
          + "(true: loc: {:.2f}, scale: {:.2f}".format(mu, sigma))
ax.plot(x[m], y[m], ls="", marker="o", c="C1", label="Data points class 1")
ax.plot(x[~m], y[~m], ls="", marker="o", c="C0", label="Data points class 0")
ax.plot(x_plot, y_plot, c="C7", lw=3, ls="--", label="True model")
ax.plot(x_plot, y_plot_est, c="C3", lw=3, label="Fitted model")
ax.axvline(
    decision_boundary, 0, 1, c="k", lw=2, ls="--", label="Class boundary")

ax.set_xlabel("x")
ax.set_ylabel("class probability")
ax.legend()

plt.savefig("../build/assets/02-img-logistic_regression.png", dpi=200)
# plt.show()
