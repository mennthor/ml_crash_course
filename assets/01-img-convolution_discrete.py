# coding: utf-8
"""
Example plots for discrete convolution.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.signal as scs


def pad_and_mirror(arr, pad_front=0, pad_back=0, mirror=False):
    if mirror:
        arr = np.atleast_1d(arr)[::-1]
    out = np.r_[pad_front * [0], arr, pad_back * [0]]
    return out


# Convolve it
signal = np.array([1., 2., 1.])
kernel = 2.**(-np.arange(-1, 2))  # Exp response
conv = scs.convolve(signal, kernel, mode="full")

# Compute all products signal * mirrored_kernel
#  shape = len(signal) + (len(kern) - 1), len(signal) + 2 * (len(kern) - 1)
_PAD = len(kernel) - 1
len_conv = len(signal) + _PAD
assert len(conv) == len_conv
signal_pad = pad_and_mirror(signal, pad_front=_PAD, pad_back=_PAD)
signal_pad_idx = np.arange(len(signal_pad)) - _PAD
prods = np.vstack(len_conv * [signal_pad])
shifted_kernels = np.empty_like(prods)
for i in range(len_conv):  # Shift kernel over padded signal
    # Make temp padded and mirrored kernel
    _pb = len_conv - 1 - i
    assert i + _pb + len(kernel) == len(signal_pad)
    shifted_kernels[i] = pad_and_mirror(
        kernel, pad_front=i, pad_back=_pb, mirror=True)
    # Multiply to padded signal
    prods[i] = prods[i] * shifted_kernels[i]

# Check if convolution is OK
my_conv = np.sum(prods, axis=1)
assert np.allclose(my_conv, conv)

# Make plot index matche signal_pad_idx
conv_pad = pad_and_mirror(conv, pad_front=_PAD)


# Main plot: vertical panel with signal, kernel, convolution
fig, (axt, axc, axb) = plt.subplots(
        3, 1, figsize=(4.5, 8), sharex=True, sharey=False)
titles = ["Signal", "Kernel (not flipped)", "Convolution"]
# Top: Signal
axt.plot(np.arange(len(signal)), signal, marker="o", color="C0", ls="")
axt.vlines(np.arange(len(signal)), 0, signal, color="C0", lw=3)
# Middle: Kernel
axc.plot(np.arange(len(kernel)), kernel, marker="o", color="C1", ls="")
axc.vlines(np.arange(len(kernel)), 0, kernel, color="C1", lw=3)
# Bottom: Convolution
axb.plot(np.arange(len(conv)), conv, marker="o", color="C7", ls="")
axb.vlines(np.arange(len(conv)), 0, conv, color="C7", lw=3)
for i, axi in enumerate((axt, axc, axb)):
    # axi.set_ylim(0, 1 + np.max(conv))
    # Only int ticks (stackoverflow.com/questions/30914462)
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    axi.yaxis.set_major_locator(MaxNLocator(integer=True))
    axi.set_title(titles[i])
    axi.grid()
axc.set_xlabel("Index $k$")
axb.set_xlabel("Index $i$")
plt.tight_layout()

plt.savefig("../build/assets/01-img-convolution_discrete.png", dpi=200)
# plt.show()


# Secondary plot: Two horizontal panels with sig + kern left and conv right
postfix = ["B", "C"]
for i, xi in enumerate([0, 2]):  # Show two examples indices
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(16, 4.5), sharey=True)

    # Plot func and kernel (mirrored)
    axl.plot(signal_pad_idx, signal_pad, label="Signal $f[k]$",
             marker="o", color="C0", ls="")
    axl.vlines(signal_pad_idx, 0, signal_pad, color="C0", lw=3)
    # Shift & mirror & offet kernel position
    axl.plot(signal_pad_idx + 0.1, shifted_kernels[xi], marker="o",
             label="Kernel $g[i - k]$", color="C1", ls="")
    axl.vlines(signal_pad_idx + 0.1, 0, shifted_kernels[xi], color="C1", lw=3)

    # Plot product (summand) at single xi
    axl.plot(signal_pad_idx + 0.2, prods[xi], marker="o",
             label="Product $f[k]\, g[i - k]$", ls="", color="C3")
    axl.vlines(signal_pad_idx + 0.2, 0, prods[xi], color="C3", lw=3)
    axl.axvline(xi, 0, 1, c="k", ls="--", lw=2)

    # Plot whole convolution function on the right plot
    _m = signal_pad_idx != xi
    axr.plot(signal_pad_idx[_m], conv_pad[_m], lw=3, c="C0", marker="o", ls="")
    axr.vlines(signal_pad_idx[_m], 0, conv_pad[_m], lw=3, color="C0")
    axr.plot(signal_pad_idx[~_m], conv_pad[~_m],  # Green where xi is
             lw=3, c="C2", marker="o", ls="", label="$x_i={}$".format(xi))
    axr.vlines(signal_pad_idx[~_m], 0, conv_pad[~_m], lw=3, color="C2")

    # Shade the areas < 0 because it's not really used in the calculations
    for axi in (axl, axr):
        axi.fill_between(signal_pad_idx, 0, 1, where=signal_pad_idx <= 0,
                         color="C7", alpha=0.25, zorder=10,
                         transform=axi.get_xaxis_transform())
        axi.set_xlim(signal_pad_idx[0], None)

    # Annotate
    axl.set_title("For $x_i = {}$".format(xi))
    axl.set_xlabel("Index $k$")
    axl.legend()
    axl.grid()

    axr.set_title("Full convolution function $(f\,\, *g)[i]$")
    axr.set_xlabel("Index $i$")
    axr.legend()
    axr.grid()

    fig.tight_layout()
    plt.savefig("../build/assets/01-img-convolution_discrete_{}.png".format(
        postfix[i]), dpi=200)
    # plt.show()
