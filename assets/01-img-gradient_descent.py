"""
Plots  good vs bad learning rate for standard gradient descent.
"""
import numpy as np
import matplotlib.pyplot as plt

from helper_gradient_descent import grad_desc_momentum


def quadratic(params, weights, bias):
    """
    Quadratic loss function: 0.5 * x^T W x + b^T x.
    Gradient: x^T W + b
    True minimum at: b W^(-1)
    """
    params = np.array(params)  # shape (2,)
    weights = np.atleast_2d(weights)  # shape (2, 2)
    bias = np.array(bias)  # shape (2,)
    if (params.shape != (2,) or weights.shape != (2, 2) or bias.shape != (2,)):
        raise ValueError("Shape mismatch in input values.")
    func = 0.5 * np.sum(params * np.sum(
        weights * params, axis=1), axis=0) + np.sum(bias * params, axis=0)
    grad = np.sum(params * weights.T, axis=1) + bias
    return func, grad


def make_values(lrate, adaptive=False):
    """ Helper to generate plot values """
    # Setup function
    weights = np.array([[2., 1.], [1., 20.]])
    bias = np.array([-5., -3])
    loss = lambda x: quadratic(x, weights, bias)
    true_min = np.sum(-bias * np.linalg.inv(weights).T, axis=1)

    # Do GD
    theta0 = [-3., -1.]
    nsteps = 100
    params, values, gradients = grad_desc_momentum(
        loss, theta0, lrate=lrate, mu=0, nsteps=nsteps, adaptive=adaptive)

    # Generate loss function surface
    xmin, xmax = -4, 4
    ymin, ymax = -2, 2
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    XX, YY = np.meshgrid(x, y)
    xx, yy = map(np.ravel, [XX, YY])

    zz = np.array([loss([xi, yi])[0] for xi, yi in zip(xx, yy)])
    ZZ = zz.reshape(XX.shape)

    # How many steps to convergence?
    eps = 1e-3
    m = values - loss(true_min)[0] < eps
    converged_idx = "Not converged within 100 steps"
    if np.any(m):
        converged_idx = "{} steps to convergence".format(np.where(m)[0][0])

    return true_min, params, XX, YY, ZZ, converged_idx


# Main: Constant good vs too low
lrate_good, lrate_low = 0.085, 0.01
true_min, params_good, XX, YY, ZZ_good, converged_good = make_values(lrate_good)
_, params_low, _, _, ZZ_low, converged_low = make_values(lrate_low)

# Plot it
fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
# Left: Good learning rate
axl.contourf(XX, YY, ZZ_good, cmap="magma")
axl.contour(XX, YY, ZZ_good, colors="w", alpha=0.5)

axl.plot(params_good[:, 0], params_good[:, 1], c="C0", marker="o")
axl.plot(true_min[0], true_min[1], marker="o", c="w")

axl.set_xlabel("$x_1$")
axl.set_ylabel("$x_2$")
axl.set_title(
    "Good learning rate ({:.3f}): {}".format(lrate_good, converged_good))

# Right: Bad learning rate
axr.contourf(XX, YY, ZZ_low, cmap="magma")
axr.contour(XX, YY, ZZ_low, colors="w", alpha=0.5)

axr.plot(params_low[:, 0], params_low[:, 1], c="C0", marker="o")
axr.plot(true_min[0], true_min[1], marker="o", c="w")

axr.set_title(
    "Too low learning rate ({:.2f}): {}".format(lrate_low, converged_low))

for axi in (axl, axr):
    axi.set_xlabel("$x_1$")
    if axi == axl:
        axi.set_ylabel("$x_2$")
    axi.set_xlim(-4, 4)
    axi.set_ylim(-2, 2)

fig.tight_layout()
plt.savefig("../build/assets/01-img-gradient_descent.png", dpi=200)
# plt.show()


# Secondary plot: Too high vs adaptive
lrate_hig, lrate_ada = 0.1, 0.1
true_min, params_hig, XX, YY, ZZ_hig, converged_hig = make_values(lrate_hig)
_, params_ada, _, _, ZZ_ada, converged_ada = make_values(
    lrate_ada, adaptive=True)

# Plot it
fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
# Left: Good learning rate
axl.contourf(XX, YY, ZZ_hig, cmap="magma")
axl.contour(XX, YY, ZZ_hig, colors="w", alpha=0.5)

axl.plot(params_hig[:, 0], params_hig[:, 1], c="C0", marker="o")
axl.plot(true_min[0], true_min[1], marker="o", c="w")

axl.set_xlabel("$x_1$")
axl.set_ylabel("$x_2$")
axl.set_title(
    "Too high learning rate ({:.1f}): {}".format(lrate_hig, converged_hig))

# Right: Bad learning rate
axr.contourf(XX, YY, ZZ_ada, cmap="magma")
axr.contour(XX, YY, ZZ_ada, colors="w", alpha=0.5)

axr.plot(params_ada[:, 0], params_ada[:, 1], c="C0", marker="o")
axr.plot(true_min[0], true_min[1], marker="o", c="w")

axr.set_title(
    "Adaptive learning rate ({:.1f}): {}".format(lrate_ada, converged_ada))

for axi in (axl, axr):
    axi.set_xlabel("$x_1$")
    if axi == axl:
        axi.set_ylabel("$x_2$")
    axi.set_xlim(-4, 4)
    axi.set_ylim(-2, 2)

fig.tight_layout()
plt.savefig("../build/assets/01-img-gradient_descent_B.png", dpi=200)
# plt.show()
