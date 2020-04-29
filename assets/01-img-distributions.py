"""
Plots PDF and CDF, discrete and continuous
"""
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 2, figsize=(8, 4.5), sharex=False)

# Discrete
pdf = np.array([1., 2., 0.5, 0.25])
pdf = pdf / np.sum(pdf)
x = np.arange(len(pdf))
cdf = np.cumsum(pdf)
cdf = np.r_[0, cdf, cdf[-1]]

# Top left: Discrete PDF
ax[0, 0].plot(x, pdf, marker="o", ls="", c="C0")
ax[0, 0].vlines(x, 0, pdf, ls="-", lw=3, color="C0")
ax[0, 0].set_ylim(0, 1.1 * np.amax(pdf))
ax[0, 0].set_title("Discrete pmf and cdf")

# Bottom left: Discrete CDF
x = np.arange(len(x) + 2) - 1
for i, yi in enumerate(cdf[:-1]):
    ax[1, 0].plot([x[i], x[i + 1]], [yi, yi], ls="-", lw=3, color="C0")
    if i > 0:
        ax[1, 0].plot(x[i], yi, ls="",
                      marker="o", fillstyle="full", lw=3, color="C0")
    if i < len(cdf) - 2:
        ax[1, 0].plot(x[i + 1], yi, markerfacecolor="w", ls="", marker="o",
                      fillstyle="none", lw=3, color="C0", markeredgewidth=2)
ax[1, 0].axhline(1, 0, 1, ls="--", c="k", lw=1)
ax[0, 0].set_xlim(x[0], x[-1])
ax[1, 0].set_xlim(x[0], x[-1])
ax[1, 0].set_ylim(0, 1.1)


# Continous
def cdf(x):
    """ Integral pdf = -x**3 / 3. + 2. * x """
    norm = 10. / 3.  # CDF(1) - CDF(-1)
    x = np.atleast_1d(x)
    out = np.zeros_like(x)
    m1 = x > 1
    m = ~np.logical_or(x < -1, m1)
    out[m] = (-x[m]**3 / 3. + 2. * x[m] - (1. / 3. - 2.)) / norm
    out[m1] = 1.
    return out


def pdf(x):
    """ -x**2 + 2 in [-1, 1] """
    norm = 10. / 3.  # CDF(1) - CDF(-1)
    x = np.atleast_1d(x)
    out = np.zeros_like(x)
    m = ~np.logical_or(x < -1, x > 1)
    out[m] = (-x[m]**2 + 2.) / norm
    return out


x_pdf = np.linspace(-1, 1, 500)
x = np.linspace(-1.25, 1.25, 500)
pdf_vals = pdf(x_pdf)
cdf_vals = cdf(x)

# Top right: Continuous PDF
ax[0, 1].plot(x_pdf, pdf_vals, c="C0", lw=3)
ax[0, 1].set_xlim(x[0], x[-1])
ax[0, 1].set_ylim(0, 1.1 * np.amax(pdf_vals))
ax[0, 1].axvline(-1, 0, 1, ls="--", c="k", lw=1)
ax[0, 1].axvline(1, 0, 1, ls="--", c="k", lw=1)
ax[0, 1].set_title("Continuous pdf and cdf")

# Bottom right: Continuous CDF
ax[1, 1].plot(x, cdf_vals, c="C0", lw=3)
ax[1, 1].axhline(1, 0, 1, ls="--", c="k", lw=1)
ax[1, 1].axvline(-1, 0, 1, ls="--", c="k", lw=1)
ax[1, 1].axvline(1, 0, 1, ls="--", c="k", lw=1)
ax[1, 1].set_xlim(x[0], x[-1])
ax[1, 1].set_ylim(0, 1.1)


fig.tight_layout()
plt.savefig("../build/assets/01-img-distributions.png", dpi=200)
# plt.show()
