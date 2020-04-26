import numpy as np


def grad_desc_momentum(func, theta0, lrate=0.001, mu=0, nsteps=100, adaptive=False):
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
    adaptive : bool
        If True, uses simple backtracking to adapt the step size.  At each
        step, the original lrate is used. But if the function does not decrease,
        it is halfed for at most 10 steps until the new function value
        decreases. (default: False)

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

    theta = np.atleast_1d(theta0)
    val, grad = func(theta)
    params, values, gradients = [theta, theta], [val], [grad]
    nadaptive_max = 1  # Non adative lrate
    if adaptive:
        nadaptive_max = 10  # Max tries for adaptive step size to avoid inf loop
    for _ in range(nsteps):
        _lrate = lrate
        _old_val = val
        for _ in range(nadaptive_max):
            # Momentum update: momentum = last step taken = lrate * grad[i-1]
            momentum = params[-1] - params[-2]
            # Gradient descent step
            theta = params[-1] - _lrate * grad + mu * momentum

            val, grad = func(theta)
            if adaptive:
                # Check if func value descreased
                if val < _old_val:
                    break
                else:
                    _lrate /= 2.  # Just half it and try again

        params.append(theta)
        values.append(val)
        gradients.append(grad)

    # Delete double first helper entry before returning
    return np.array(params[1:]), np.squeeze(values), np.atleast_2d(gradients)
