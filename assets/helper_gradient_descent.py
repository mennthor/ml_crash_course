import numpy as np


def grad_desc_momentum(func, theta0, lrate=0.001, mu=0, nsteps=100):
    """
    Parameters
    ----------
    func : callable
        Takes parameter input vector as single argument.
        Must return a tuple of (single function value, gradient vector).
    theta0 : array-like, shape (N,)
        The initial parameter set given as ``func(theta0)`.
    lrate : float
        Learning rate >0. (default: 0.001)
    mu : float
        Momentum strength in [0, 1], if 0 we have standard GD. (default: 0)
    nsteps : int
        Number of gradient steps. (default: 100)

    Returns
    -------
    params : array-like
        Parameter values per step.
    values : array-like
        Function values per step.
    gradients : array-like
        Gradient vectors per step.
    """
    if lrate < 0:
        raise ValueError("lrate must be > 0")
    if mu < 0 or mu > 1:
        raise ValueError("mu must be in [0, 1]")
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")

    params, values, gradients = [], [], []
    theta = np.atleast_1d(theta0)
    last_step = theta  # Initial momentum is zero with this choice
    for i in range(nsteps):
        params.append(theta)

        # Get new gradients
        val, grad = func(theta)
        # Momentum update
        dparams = theta - last_step  # = lrate * grad[i-1]
        last_step = theta
        # Gradient descent step
        theta = theta - lrate * grad + mu * dparams

        values.append(val)
        gradients.append(grad)
    return np.array(params), np.squeeze(values), np.atleast_2d(gradients)
