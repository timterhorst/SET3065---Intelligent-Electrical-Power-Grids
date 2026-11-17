"""Logistic regression with Adam for the Secondary Mushroom dataset (NLO1)."""

import numpy as np
import pandas as pd


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
    data = pd.read_csv(path, sep=";")
    class_col = data.columns[0]
    columns = []

    for c in data.columns[1:]:
        column = data[c]
        # Numeric if every value converts to float; otherwise treat as categorical.
        try:
            values = column.astype(float).to_numpy()
            columns.append(values)
        except (ValueError, TypeError):
            values = column.astype(str).to_numpy()
            unique_values = sorted(set(values))
            for v in unique_values[1:]:
                columns.append((values == v).astype(float))

    m = data.shape[0]
    X = np.column_stack(columns).astype(float)
    y = (data[class_col].astype(str).str.strip() == "poisonous").to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    return X[perm], y[perm]


def _softplus(t):
    """Numerically stable softplus: log(1 + exp(t))."""
    return np.logaddexp(0.0, np.asarray(t, dtype=float))


def _sigmoid(t):
    """Numerically stable logistic sigmoid Φ(t)."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def func_llh(theta, X, y):
    """Logistic log-likelihood F (equation 3).

    F = sum_j (y_j * t_j - log(1 + exp(t_j))) with t_j = w0 + w · x_j
    and theta = [w0; w]. CodeGrade expects the sum of log-likelihoods
    (equivalent to (1/m) * sum F_j with F_j = m * log ell_j).

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
        Total log-likelihood sum_j log ell_j.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    t = theta[0] + X @ theta[1:]
    return float(np.sum(y * t - _softplus(t)))


def mini_batch_grad_llh(theta, X, y, batch_bounds):
    """Mini-batch gradient of the average log-likelihood F (equation 4).

    For batch B given by ``batch_bounds = (b_start, b_end)`` (0-based, with
    negative indices counting from the end / wrapping), returns
    (m / |B|) * Xb_aug.T @ (yb - Φ(tb)).

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
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = X.shape[0]
    b_start, b_end = batch_bounds
    idx = np.arange(b_start, b_end) % m
    Xb = X[idx]
    yb = y[idx]
    batch_size = idx.size

    t = theta[0] + Xb @ theta[1:]
    resid = yb - _sigmoid(t)
    grad = np.empty(theta.size, dtype=float)
    grad[0] = np.sum(resid)
    grad[1:] = Xb.T @ resid
    return (m / batch_size) * grad


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
    if 2 * b_end - b_start < num_obs:
        return (b_end, 2 * b_end - b_start)
    return (b_end - num_obs, 2 * b_end - b_start - num_obs)


def adam(
    theta0,
    func,
    mini_batch_grad,
    num_obs,
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
    theta = np.asarray(theta0, dtype=float).reshape(-1).copy()
    thetas = [theta.copy()]
    m_mom = np.zeros_like(theta)
    v_mom = np.zeros_like(theta)

    bounds = (0, min(num_obs, batch_size))
    last_f = np.inf
    k = 1
    while True:
        g = np.asarray(mini_batch_grad(theta, bounds), dtype=float).reshape(-1)
        m_mom = beta1 * m_mom + (1.0 - beta1) * g
        v_mom = beta2 * v_mom + (1.0 - beta2) * (g * g)
        mhat = m_mom / (1.0 - beta1**k)
        vhat = v_mom / (1.0 - beta2**k)
        theta = theta - alpha * mhat / (np.sqrt(vhat) + epsilon)
        thetas.append(theta.copy())

        next_bounds = next_batch_bounds(bounds, num_obs)
        if next_bounds[0] < bounds[1]:
            f_new = float(func(theta))
            if last_f - f_new < delta:
                return thetas
            last_f = f_new

        bounds = next_bounds
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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta0 = np.zeros(X.shape[1] + 1, dtype=float)

    def f(theta):
        return -func_llh(theta, X, y)

    def g(theta, batch_bounds):
        return -mini_batch_grad_llh(theta, X, y, batch_bounds)

    thetas = adam(theta0, f, g, X.shape[0])
    return thetas[-1]


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
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    t = theta[0] + X @ theta[1:]
    return (t >= 0.0).astype(float)


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
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    return float(np.mean(y == yhat))
