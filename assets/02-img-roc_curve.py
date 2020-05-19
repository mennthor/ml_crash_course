#!/usr/bin/env python
# coding: utf-8

"""
Example ROC curve for logistic regressor
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

plt.rcParams["lines.linewidth"] = 3
rng = np.random.RandomState(1337)
# Make 1D regression and 2D classification data.

# Classification truth: Separated by curved region
# Sample in box and set class by comapring to quadratic line
xmin, xmax = -3, 3
xc = np.sort(rng.uniform(xmin, xmax, size=75))
yc = rng.uniform(xmin, xmax, size=75)
comp = 0.5 * yc**2 - 2  # True class boundary
m = xc > comp
classes = np.zeros_like(xc)
classes[m] = 1
# Now smear stuff out a little bit
xc += rng.uniform(-0.5, 0.5, size=len(xc))
yc += rng.uniform(-1., 1., size=len(yc))

# Classification
# Underfit with logistic regression
data = np.vstack([xc, yc]).T
logreg = LogisticRegression(random_state=rng)
logreg.fit(data, classes)

# Get ROC and AUC for train data
preds = logreg.predict_proba(data)
fpr, tpr, threshs = roc_curve(classes, preds[:, 1], pos_label=1)
# Stepwise integral (summing rectangles)
auc = np.sum(np.diff(fpr) * tpr[:-1])

# Plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(fpr, tpr, label="ROC")
ax.fill_between(fpr, 0, tpr, color="C0", alpha=0.1, label="AUC")
# Guessing classifier for comparison
ax.plot([0, 1], [0, 1], ls="--", c="C7", label="Guessing")

ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Receiver operating characteristic (ROC) curve")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid()
ax.legend(loc="lower right")

fig.tight_layout()
plt.savefig("../build/assets/02-img-roc_curve.png", dpi=200)
# plt.show()
