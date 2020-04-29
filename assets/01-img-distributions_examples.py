"""
Plots PDF and CDF, discrete and continuous
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs


fig, (axt, axc, axb) = plt.subplots(3, 1, figsize=(9, 5))

# Top: Gaussians
x = np.linspace(-5, 15, 500)
means = [0, 2, 5]
stddevs = [1, 0.5, 3]
for mi, si in zip(means, stddevs):
    pdf = scs.norm.pdf(x, mi, si)
    axt.plot(x, pdf, lw=3, label="$\mathcal{{N}}({}, {})$".format(mi, si))
axt.grid()
axt.set_xlim(x[0], x[-1])
axt.set_ylim(0, None)
axt.legend()
axt.set_xlabel("$x$")
axt.set_ylabel("pdf")

# Center: Binomials
n = 20
k = np.arange(n + 1)
ps = [0.2, 0.5, 0.9]
for pi in ps:
    pmf = scs.binom.pmf(k, n, pi)
    axc.plot(k, pmf, drawstyle="steps-pre", lw=3,
             label="$\mathcal{{B}}({}, {})$".format(n, pi))
axc.grid()
axc.set_xlim(k[0], k[-1])
axc.set_ylim(0, None)
axc.legend(ncol=3)
axc.set_ylabel("pmf")

# Bottom: Poissons
mus = [1., 5.5, 20.]
k = np.arange(np.ceil(np.amax(mus) + 3))
for mui in mus:
    pmf = scs.poisson.pmf(k, mui)
    axb.plot(k, pmf, drawstyle="steps-pre", lw=3,
             label="$\mathcal{{P}}({})$".format(mui))
axb.grid()
axb.set_xlim(k[0], k[-1])
axb.set_ylim(0, None)
axb.legend(ncol=3)
axb.set_xlabel("Index $k$")
axb.set_ylabel("pmf")

fig.tight_layout()
plt.savefig("../build/assets/01-img-distributions_examples.png", dpi=200)
# plt.show()
