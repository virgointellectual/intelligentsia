# Copyright (c) 2014 Steven de Rooij, Tim van Erven, Peter D. Grünwald and Wouter M. Koolen.
# 
# Author: Arnold Salas <asalas@virgointellectual.com> (Python translation)

"""Python implementation of the AdaHedge algorithm used in the context
of prediction with expert advice.

Functions
---------

adahedge()
    Return the weights and losses of AdaHedge.

"""

__all__ = ['adahedge']

import math as _math

import numpy as _np
import numpy.typing as _npt


def adahedge(l: _npt.ArrayLike) -> tuple[_np.ndarray, _np.ndarray]:
    """Return the weights and losses of AdaHedge.

    Parameters
    ----------
    l : array_like
        A T-by-K data structure containing the losses of the experts.
        The entry at position (t,k) is the loss of expert k at time t.

    Returns
    -------
    W : (T, K) ndarray
        The weights assigned by the algorithm to the K experts across
        timesteps.
    h : (T,) ndarray
        The loss incurred by the algorithm at each timestep.

    Notes
    -----
    This is the Python translation of the original, numerically robust
    MATLAB implementation of AdaHedge (see [1]_, Figure 1).

    .. [1] S. de Rooij et al., "Follow the Leader If You Can, Hedge If
       You Must", Journal of Machine Learning Research, vol. 15,
       pp. 1281-1316, 2014.
       URL: <https://jmlr.org/papers/volume15/rooij14a/rooij14a.pdf>.
    """
    l = _np.asarray(l)
    T, K = l.shape
    W = _np.empty((T, K))
    h = _np.empty(T)
    L = _np.zeros(K)
    Delta = 0

    for t in range(T):
        try:
            eta = _math.log(K) / Delta
        except ZeroDivisionError:
            eta = _math.inf
        w, Mprev = _mix(eta, L)
        W[t] = w
        h[t] = _np.dot(w, l[t])
        L += l[t]
        _, M = _mix(eta, L)
        delta = max(0, h[t] - (M-Mprev))
        # Max clips numeric Jensen violation.
        Delta += delta

    return W, h


def _mix(eta, L):
    """Return the posterior weights and mix loss, avoiding numerical
    instability.

    Parameters
    ----------
    eta : float
        The learning rate (must be positive).
    L : (K,) ndarray
        The cumulative losses of the K experts at a given timestep.

    Returns
    -------
    w : (K,) ndarray
        The weights assigned to the K experts at the timestep
        corresponding to `L`.
    M : float
        The mix loss at the timestep associated with `L`.

    Notes
    -----
    This is the Python implementation of the mix loss in [1]_, Figure 1.

    .. [1] S. de Rooij et al., "Follow the Leader If You Can, Hedge If
       You Must", Journal of Machine Learning Research, vol. 15,
       pp. 1281-1316, 2014.
       URL: <https://jmlr.org/papers/volume15/rooij14a/rooij14a.pdf>.
    """
    mn = L.min()
    if eta == _math.inf:  # Limit behaviour: FTL.
        w = (L == mn).astype(_np.float64)
    else:
        w = _np.exp(-eta * (L-mn))
    s = w.sum()
    w /= s
    M = mn - _math.log(s/len(L))/eta
    return w, M
