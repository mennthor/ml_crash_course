"""
Plots functions and their derviative
"""
import numpy as np
import matplotlib.pyplot as plt


g1 = lambda x: np.sin(2 * x) + x**2
g2 = lambda x: 2. / (1. + np.exp(-x)) - 1.

dg1 = lambda x: np.cos(2 * x) * 2 + 2 * x
dg2 = lambda x: 2. / (1. + np.exp(-x))**2 * np.exp(-x)

x1 = np.linspace(-1, 3, 100)
x2 = np.linspace(-5, 5, 100)
val1 = g1(x1)
val2 = g2(x2)
grad1 = dg1(x1)
grad2 = dg2(x2)

grad_num1 = np.gradient(val1, x1)
grad_num2 = np.gradient(val2, x2)

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 3))
axl.plot(x1, val1, label="f(x)", c="C0", lw=3)
axl.plot(x1, grad1, label="f'(x)", c="C1", lw=3)
axl.plot(x1, grad_num1, label="f'(x) numeric", c="k", ls=":", lw=3)
axl.axvline(0, 0, 1, lw=1, c="k")
axl.axhline(0, 0, 1, lw=1, c="k")
axl.set_title("$\sin(2x) + x^2$")
axl.grid()
axl.legend()

axr.plot(x2, val2, label="f(x)", c="C0", lw=3)
axr.plot(x2, grad2, label="f'(x)", c="C1", lw=3)
axr.plot(x2, grad_num2, label="f'(x) numeric", c="k", ls=":", lw=3)
axr.axvline(0, 0, 1, lw=1, c="k")
axr.axhline(0, 0, 1, lw=1, c="k")
axr.set_title("$2 / (1+e^{-x}) - 1$")
axr.grid()
axr.legend()

for axi in (axl, axr):
    axi.set_xlabel("x")
    axi.set_ylabel("y")

fig.tight_layout()
plt.savefig("../build/assets/01-img-derivatives.png", dpi=200)
# plt.show()
