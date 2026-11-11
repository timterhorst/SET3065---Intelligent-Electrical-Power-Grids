"""Logistic regression with Adam for the Secondary Mushroom dataset (NLO1)."""

from __future__ import annotations

import numpy as np


def load_data(path, seed):
    """Load and encode the Secondary Mushroom CSV into design matrix X and labels y.

    Reads a semicolon-separated CSV with a header row. The first column is the
    class label (edible/poisonous); remaining columns are features. Numeric
    features are kept as float columns. Categorical features are one-hot encoded
    with unique levels sorted alphabetically and the first level dropped.
    Labels are 1 for poisonous and 0 otherwise. Rows of X and entries of y are
    shuffled with a NumPy Generator seeded by ``seed``.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    seed : int
        Seed for the shuffle RNG.

    Returns
    -------
    X : ndarray, shape (m, p)
        Float feature matrix (ndim == 2).
    y : ndarray, shape (m,)
        Float labels in {0, 1} (ndim == 1).
    """
    # Prefer a light dependency footprint: parse with the csv module via numpy/pandas.
    try:
        import pandas as pd

        df = pd.read_csv(path, sep=";")
        col_names = list(df.columns)
        raw = df.to_numpy()
    except Exception:
        # Fallback without pandas: first row is header.
        data = np.genfromtxt(path, delimiter=";", dtype=str, encoding="utf-8")
        if data.ndim != 2:
            raise ValueError("CSV must be two-dimensional after loading")
        col_names = [c.strip() for c in data[0].tolist()]
        raw = data[1:]

    m = raw.shape[0]
    class_col = raw[:, 0].astype(str)
    y = (np.char.lower(class_col) == "poisonous").astype(np.float64)

    feature_blocks = []
    for j, name in enumerate(col_names[1:], start=1):
        col = raw[:, j]
        numeric = False
        known_numeric = name in ("cap-diameter", "stem-height", "stem-width")
        if known_numeric:
            numeric = True
            values = col.astype(np.float64)
        else:
            try:
                values = np.asarray(col, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("non-finite numeric parse")
                numeric = True
            except (ValueError, TypeError):
                numeric = False

        if numeric:
            feature_blocks.append(values.reshape(m, 1))
        else:
            levels = sorted({str(v) for v in col})
            # Drop the first alphabetical level (reference category).
            for level in levels[1:]:
                feature_blocks.append((col.astype(str) == level).astype(np.float64).reshape(m, 1))

    if feature_blocks:
        X = np.hstack(feature_blocks).astype(np.float64)
    else:
        X = np.zeros((m, 0), dtype=np.float64)

    if X.ndim != 2:
        X = np.atleast_2d(X)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.ndim != 1:
        y = y.ravel()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    X = X[perm]
    y = y[perm]
    return X, y


def _softplus(t):
    """Numerically stable softplus: log(1 + exp(t))."""
    t = np.asarray(t, dtype=np.float64)
    return np.logaddexp(0.0, t)


def _sigmoid(t):
    """Numerically stable logistic sigmoid Φ(t)."""
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t, dtype=np.float64)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def _linear_scores(theta, X):
    """Compute t = w0 + X @ w for theta = [w0; w]."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    w0 = theta[0]
    w = theta[1:]
    return w0 + X @ w


def func_llh(theta, X, y):
    """Average logistic log-likelihood F (equation 3).

    F = mean_j (y_j * t_j - log(1 + exp(t_j))) with t_j = w0 + w · x_j
    and theta = [w0; w].

    Parameters
    ----------
    theta : array_like, shape (p + 1,)
        Intercept and weights.
    X : array_like, shape (m, p)
        Feature matrix.
    y : array_like, shape (m,)
        Labels in {0, 1}.

    Returns
    -------
    float
        Average log-likelihood.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    t = _linear_scores(theta, X)
    # y*t - softplus(t); softplus is stable log(1+exp(t))
    vals = y * t - _softplus(t)
    return float(np.mean(vals))


def _batch_indices(batch_bounds, m):
    """Map half-open batch_bounds into row indices, including wrap-around."""
    b_start, b_end = batch_bounds
    if not (-m <= b_start < b_end < m):
        # Still attempt to build indices; keep behaviour useful for Adam.
        pass
    # Indices in range(b_start, b_end) modulo m (handles negative windows).
    return np.arange(b_start, b_end) % m


def mini_batch_grad_llh(theta, X, y, batch_bounds):
    """Mini-batch gradient of the average log-likelihood F (equation 4).

    For batch B given by ``batch_bounds = (b_start, b_end)`` (0-based, with
    negative indices counting from the end / wrapping), returns
    (m / |B|) * Xb_aug.T @ (yb - Φ(tb)), which equals the average over B of
    ∇F_j with ∇F_j = m * (y_j - Φ(t_j)) * [1; x_j].

    Parameters
    ----------
    theta : array_like, shape (p + 1,)
        Intercept and weights.
    X : array_like, shape (m, p)
        Feature matrix.
    y : array_like, shape (m,)
        Labels in {0, 1}.
    batch_bounds : tuple of int
        (b_start, b_end) with -m ≤ b_start < b_end < m (wrap-around allowed).

    Returns
    -------
    ndarray, shape (p + 1,), ndim == 1
        Mini-batch gradient.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = X.shape[0]
    idx = _batch_indices(batch_bounds, m)
    Xb = X[idx]
    yb = y[idx]
    batch_size = idx.size
    if batch_size == 0:
        return np.zeros(X.shape[1] + 1, dtype=np.float64)

    tb = _linear_scores(theta, Xb)
    resid = yb - _sigmoid(tb)
    # Augmented design: [1, Xb]
    ones = np.ones((batch_size, 1), dtype=np.float64)
    Xb_aug = np.hstack([ones, Xb])
    grad = (m / batch_size) * (Xb_aug.T @ resid)
    return np.asarray(grad, dtype=np.float64).reshape(-1)


def next_batch_bounds(batch_bounds, num_obs):
    """Advance mini-batch bounds by one window, wrapping when a pass ends.

    If ``2 * b_end - b_start < m``, returns ``(b_end, 2 * b_end - b_start)``;
    otherwise returns ``(b_end - m, 2 * b_end - b_start - m)``.

    Parameters
    ----------
    batch_bounds : tuple of int
        Current (b_start, b_end).
    num_obs : int
        Number of observations m.

    Returns
    -------
    tuple of int
        Next (b_start, b_end).
    """
    b_start, b_end = batch_bounds
    m = num_obs
    if 2 * b_end - b_start < m:
        return (b_end, 2 * b_end - b_start)
    return (b_end - m, 2 * b_end - b_start - m)


def adam(
    theta0,
    func,
    mini_batch_grad,
    num_obs,
    *,
    batch_size=128,
    alpha=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    delta=1e-8,
):
    """Adam optimizer for minimizing ``func`` with mini-batch gradients.

    Implements Algorithm 2: bias-corrected Adam updates over successive
    mini-batches. After each completed pass over the data (detected when the
    next batch start is before the current batch end), evaluates ``func``.
    Stops when the improvement ``last_f - f_new`` is smaller than ``delta``.

    Parameters
    ----------
    theta0 : array_like
        Starting parameter vector.
    func : callable
        Objective ``func(theta)`` to minimize.
    mini_batch_grad : callable
        Gradient ``mini_batch_grad(theta, batch_bounds)``.
    num_obs : int
        Number of observations m.
    batch_size : int, optional
        Mini-batch size (default 128).
    alpha : float, optional
        Step size (default 0.001).
    beta1, beta2 : float, optional
        Exponential decay rates for moment estimates.
    epsilon : float, optional
        Numerical stabilizer in the Adam update.
    delta : float, optional
        Minimum required improvement in ``func`` per pass.

    Returns
    -------
    list of ndarray
        Iterates ``[theta0, theta1, ..., theta_k]`` including the start point.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1).copy()
    thetas = [theta.copy()]
    m_mom = np.zeros_like(theta)
    v_mom = np.zeros_like(theta)

    b_start = 0
    b_end = min(num_obs, batch_size)
    last_f = np.inf
    k = 1
    while True:
        g = np.asarray(mini_batch_grad(theta, (b_start, b_end)), dtype=np.float64).reshape(-1)
        m_mom = beta1 * m_mom + (1.0 - beta1) * g
        v_mom = beta2 * v_mom + (1.0 - beta2) * (g * g)
        mhat = m_mom / (1.0 - beta1**k)
        vhat = v_mom / (1.0 - beta2**k)
        theta = theta - alpha * mhat / (np.sqrt(vhat) + epsilon)
        thetas.append(theta.copy())

        next_b_start, next_b_end = next_batch_bounds((b_start, b_end), num_obs)
        if next_b_start < b_end:
            f_new = float(func(theta))
            if last_f - f_new < delta:
                return thetas
            last_f = f_new

        b_start, b_end = next_b_start, next_b_end
        k += 1


def logit(X, y):
    """Fit logistic regression by maximizing the average log-likelihood via Adam.

    Starts at the zero vector of length p+1 and calls ``adam`` with default
    hyperparameters. Because Adam minimizes, the objective and mini-batch
    gradient are negated wrappers around ``func_llh`` and
    ``mini_batch_grad_llh``.

    Parameters
    ----------
    X : array_like, shape (m, p)
        Feature matrix.
    y : array_like, shape (m,)
        Labels in {0, 1}.

    Returns
    -------
    ndarray, shape (p + 1,)
        Last Adam iterate (intercept and weights).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m, p = X.shape
    theta0 = np.zeros(p + 1, dtype=np.float64)

    def f(theta):
        return -func_llh(theta, X, y)

    def g(theta, bounds):
        return -mini_batch_grad_llh(theta, X, y, bounds)

    thetas = adam(theta0, f, g, m)
    return np.asarray(thetas[-1], dtype=np.float64).reshape(-1)


def logistic_prediction(X, theta):
    """Predict binary labels with the fitted logistic model.

    Returns 1 when Φ(w0 + w · x_j) ≥ 1/2 (equivalently when t_j ≥ 0), else 0.

    Parameters
    ----------
    X : array_like, shape (m, p)
        Feature matrix.
    theta : array_like, shape (p + 1,)
        Intercept and weights.

    Returns
    -------
    ndarray, shape (m,), ndim == 1
        Predicted labels as floats in {0, 1}.
    """
    t = _linear_scores(theta, X)
    yhat = (t >= 0.0).astype(np.float64)
    return yhat.reshape(-1)


def accuracy(y, yhat):
    """Fraction of matching labels between ``y`` and ``yhat``.

    Parameters
    ----------
    y : array_like, shape (m,)
        True labels.
    yhat : array_like, shape (m,)
        Predicted labels.

    Returns
    -------
    float
        Mean of (y == yhat).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    return float(np.mean(y == yhat))
