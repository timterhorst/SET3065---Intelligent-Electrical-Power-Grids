import numpy as np


def load_data(path, seed):
    """ 
    De regels in secundary_mushroom.csv corresponderen met een hypothetische steekproef van paddenstoelen uit 173 verschillende soorten. We noemen deze 
    regels de observaties. De eigenschappen van een specifieke paddenstoel worden gescheiden door puntkomma’s, en bevatten in volgorde de informatie uit Tabel 
    1 (opdracht).
    Eerste eigenschap in dataset = klasse, de andere eigenschappen in dataset = kenmerken.
    Om iets met de data te kunnen, moet er getransformeerd worden naar numerieke data, volgens Algoritme 1. 
    Bouwt x: numerieke kolommen als floats; categorische kolommen one-hot
    Paddenstoel giftig is (y = 1) of eetbaar (y = 0)
    Rijen van X en elementen van y worden geschud met een NumPy-generator met seed.
    """
    raw = np.genfromtxt(path, delimiter=";", dtype=str, encoding="utf-8")
    body = raw[1:]
    m = body.shape[0]
    columns = []

    for j in range(1, body.shape[1]):
        col = body[:, j]
        try:
            values = col.astype(float)
            columns.append(values)
        except ValueError:
            values = np.asarray(col, dtype=str)
            unique_values = sorted(set(values.tolist()))
            for v in unique_values[1:]:
                columns.append((values == v).astype(float))

    X = np.column_stack(columns).astype(float)
    y = (np.char.strip(body[:, 0]) == "poisonous").astype(float)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    return X[perm], y[perm]


def _softplus(t):
    """
    Numerieke stabiele softplus: log(1 + exp(t)).
    Voorkommt overflow bij grote t.
    """
    return np.logaddexp(0.0, np.asarray(t, dtype=float))


def _sigmoid(t):
    """
    Numerieke stabiele logistische sigmoid Φ(t) = 1/(1+e^{-t}).
    Voor t ≥ 0 en t < 0 een andere formule, zodat exp niet explodeert.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def func_llh(theta, X, y):
    """
    Gemiddelde logistische log-aannemelijkheid F (formue 3)
    F is het gemiddelde van y·t − log(1+e^t), met t = w0 + w·x
    theta: intercept w0 en gewichten w
    X: kenmerkenmatrix
    y: labels 0 of 1
    geeft het gemiddelde van de log-aannemelijkheid
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    t = theta[0] + X @ theta[1:]
    return float(np.sum(y * t - _softplus(t)))


def mini_batch_grad_llh(theta, X, y, batch_bounds):
    """
    Mini-batch gradient van F (formule 4). 
    Begin/eindindex negatieve indices vanaf het einde/wrap-around
    Geeft gradiënt = (m / batchgrootte) × …
        1D vector met de gradient
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
    """
    Schuif het batch-venster één stap --> wrap als een ronde klaar is.
    De twee formules van opdracht samenvoegen
    """
    b_start, b_end = batch_bounds
    if 2 * b_end - b_start < num_obs:
        return (b_end, 2 * b_end - b_start)
    return (b_end - num_obs, 2 * b_end - b_start - num_obs)


def adam(theta0, func, mini_batch_grad, num_obs, batch_size=128, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, delta=1e-8):
    """
    Adam-optimaliseerder die func minimaliseert met mini-batch gradiënten.
    Voert Algoritme 2 uit: Adam-updates met bias-correctie.
    Na elke volle ronde over de data: bereken func.
    Stop als de verbetering kleiner is dan delta.
    Parameters
        theta0 : startpunt
        func : doelfunctie om te minimaliseren
        mini_batch_grad : mini-batch gradiëntfunctie
        num_obs : aantal observaties m 
        batch_size : batchgrootte
        alpha : staplengte
        beta1, beta2 : vervalsnelheden voor 1e en 2e moment
        epsilon : kleine constante tegen delen door nul
        delta : minimale verbetering per ronde
    Geeft lijst van alle bezochte θ’s, inclusief start.
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
    """
    Fit logistische regressie door de gemiddelde log-aannemelijkheid te maximaliseren met Adam.
    Start bij de nulvector van lengte p+1; roept adam aan zonder eigen defaults.
    Adam minimmaliseert, dus we geven −F en −gradiënt mee.
    Geeft laatste θ (w0 en w)
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
    """
    Voorspel binaire labels met het gefitte model.
    Geeft 1 als Φ ≥ ½ (gelijk aan t ≥ 0), anders 0. 
    Geeft voorspelde labels als floats 0/1
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    t = theta[0] + X @ theta[1:]
    return (t >= 0.0).astype(float)


def accuracy(y, yhat):
    """
    Fractie gelijke labels tussen y en ŷ.
    y: echte labels
    yhat: voorspelde labes
    Geeft gemiddelde can (y == ŷ)
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    return float(np.mean(y == yhat))
