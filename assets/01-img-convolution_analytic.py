# coding: utf-8
"""
Example plots for analytic convolution in three separate panels.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import scipy.stats as scs


def kernel(t):
    """ Rectangle function in (-1, 1) with linear raising edge (-1, -0.5) """
    t = np.atleast_1d(t)
    out = np.zeros_like(t, dtype=np.float)
    # Constant rectangle
    m = np.logical_and(t >= -0.5, t < 1)
    out[m] = 0.5
    # Linear raising edge [-1, -0.5] -> [0, 0.5]
    m = np.logical_and(t > -1, t < -0.5)
    out[m] = np.maximum(0, t[m] + 1)
    return out


def func(t):
    """ Skewed gaussian with skew strength alpha """
    alpha = -6
    return 2 * scs.norm.pdf(t) * scs.norm.cdf(alpha * t)


def prod(t, xi):
    return func(t) * kernel(xi - t)
    # It's commutative, but this way seems to integrate better numerically
    # when using quad. Using simps with tight spacing is good enough here though
    # return func(xi - t) * kernel(t)


# Sampling range for the convolution function
x = np.arange(-10, 10, 0.1)
# Grid spacing for simpson integration, more stable than sci.quad
t_simps = np.arange(-20, 20, 0.01)
res = []
for xi in x:
    # res.append(sci.quad(prod, a=-20, b=20, args=xi))
    res.append(sci.simps(prod(t_simps, xi), x=t_simps))

# conv_vals = np.array([v[0] for v in res])  # Use with quad
conv_vals = np.array(res)

# Plots
postfix = ["", "_B", "_C"]
t = np.linspace(-5, 5, 200)  # Spacing for plotting
for i, xi in enumerate([-1.5, 0., 0.5]):
    idx = np.isclose(x, xi)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(16, 4.5), sharey=True)

    # Plot func and kernel (mirrored)
    axl.plot(t, func(t), label="Function $f(t)$")
    axl.plot(t, kernel(xi - t), label="Kernel $g(x_i - t)$")

    # Plot product (integrand) at single xi
    axl.plot(t, prod(t, xi), ls="--", color="C3", alpha=.5,
             label="Integrand $f(t)\, g(x_i - t)$")
    axl.fill_between(t, prod(t, xi), ls="--", color="C3", alpha=.25, hatch="/")
    axl.axvline(xi, 0, 1, c="k", ls="--", lw=2)

    # Plot whole convolution function on the right plot
    axr.plot(x, conv_vals, lw=3)
    axr.axvline(xi, 0, 1, c="k", ls=":", lw=2)
    axr.plot(xi, conv_vals[idx],
             ls="", c="k", marker="o", label="$x_i={:.1f}$".format(xi))

    # Annotate
    axl.set_title("For $x_i = {}$".format(xi))
    axl.set_xlabel("$t$")
    axl.legend()
    axl.grid()

    axr.set_title("Full convolution function $(f\,\, *g)(x)$")
    axr.set_xlabel("$x$")
    axr.legend()
    axr.grid()

    fig.tight_layout()
    plt.savefig("../build/assets/01-img-convolution_analytic{}.png".format(
        postfix[i]), dpi=200)
    # plt.show()
