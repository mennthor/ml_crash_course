# coding: utf-8
"""
Example plots for straight line fit to data.
Code from:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs


# Generate some data points on a straight line with y noise
rng = np.random.RandomState(1337)
x = np.random.uniform(0, 10, size=20)
y = 1.5 * x + 1 + np.random.normal(size=len(x))

# Perform the linear regression
slope, intercept, r_value, p_value, std_err = scs.linregress(x, y)

# Plot
x_plot = np.linspace(0, 10, 2)
plt.title("Fitted slope: {:.2f}, intercept: {:.2f}".format(slope, intercept))
plt.plot(x, y, marker="o", ls="", label="Data points")
plt.plot(x_plot, intercept + slope * x_plot,
         lw=3, c="C7", ls="--", label="Fitted line")

plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.savefig("../build/assets/02-img-fit_straight_line_example.png", dpi=200)
# plt.show()
