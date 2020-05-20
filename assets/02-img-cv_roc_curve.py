#!/usr/bin/env python
# coding: utf-8

"""
Averaged cross validated ROC curve example
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

plt.rcParams["lines.linewidth"] = 3
rng = np.random.RandomState(1337)

# Classification truth: Separated by curved region
# Sample in box and set class by comapring to quadratic line
xmin, xmax = -3, 3
ndata = 1000
xc = np.sort(rng.uniform(xmin, xmax, size=ndata))
yc = rng.uniform(xmin, xmax, size=ndata)
comp = 0.5 * yc**2 - 2  # True class boundary
m = xc > comp
classes = np.zeros_like(xc)
classes[m] = 1
# Now smear stuff out a little bit
xc += rng.uniform(-0.5, 0.5, size=len(xc))
yc += rng.uniform(-1., 1., size=len(yc))

# Cross validate, keep each model
ncvs = 10
fpr_grid = np.linspace(0, 1, 101)
data = np.vstack([xc, yc]).T
models, preds, perfs, tprs = [], [], [], []
kfold_cv = StratifiedKFold(n_splits=ncvs, random_state=rng, shuffle=True)
for train_index, test_index in kfold_cv.split(data, classes):
    data_cv_train = data[train_index]
    classes_cv_train = classes[train_index]
    logreg = LogisticRegression(random_state=rng)
    logreg.fit(data_cv_train, classes_cv_train)

    data_cv_test = data[test_index]
    classes_cv_test = classes[test_index]
    preds.append(logreg.predict_proba(data_cv_test)[:, 1])
    perfs.append(roc_curve(classes_cv_test, preds[-1], pos_label=1))

    # Store interpolated ROC values to calculate mean performance later
    interp_tpr = np.interp(fpr_grid, perfs[-1][0], perfs[-1][1])
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

tpr_mean = np.mean(tprs, axis=0)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
for i, (fpr, tpr, _) in enumerate(perfs):
    ax.plot(fpr, tpr, c="C7", alpha=0.25)

# Mean CV model
ax.plot(fpr_grid, tpr_mean, c="C3", alpha=1, label="Mean CV model")

# Guessing classifier for comparison
ax.plot([0, 1], [0, 1], ls="--", c="C7", label="Guessing")

ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("10-fold cross validated ROC curve")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid()
ax.legend(loc="lower right")

fig.tight_layout()
plt.savefig("../build/assets/02-img-cv_roc_curve.png", dpi=200)
# plt.show()
