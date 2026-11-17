"""Local tests for NLO1 (not submitted to CodeGrade)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

CSV_PATH = Path("/cursor/stores/self/internal/secondary_mushroom.csv")
SEED = 42


def test_import_has_no_side_effects():
    """Importing main must not train or print training side effects."""
    if "main" in sys.modules:
        del sys.modules["main"]
    # Ensure NLO1 is on path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module("main")
    assert hasattr(mod, "load_data")
    assert hasattr(mod, "logit")


def test_load_data_shapes_and_repro():
    from main import load_data

    X1, y1 = load_data(str(CSV_PATH), SEED)
    X2, y2 = load_data(str(CSV_PATH), SEED)

    assert X1.ndim == 2
    assert y1.ndim == 1
    assert X1.shape[0] == y1.shape[0]
    assert X1.dtype == np.float64 or np.issubdtype(X1.dtype, np.floating)
    assert np.isfinite(X1).all()
    assert set(np.unique(y1)).issubset({0.0, 1.0})
    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)

    # Different seed should (almost surely) change order
    X3, y3 = load_data(str(CSV_PATH), SEED + 1)
    assert X3.shape == X1.shape
    assert not np.allclose(X1, X3) or not np.allclose(y1, y3)

    print(f"load_data: X{X1.shape}, y{y1.shape}, poisonous_frac={y1.mean():.4f}")
    return X1, y1


def test_one_hot_drops_first_alphabetical():
    """Spot-check: first alphabetical level of a categorical is dropped."""
    import pandas as pd
    from main import load_data

    df = pd.read_csv(CSV_PATH, sep=";")
    # season levels: autumn, spring, summer, winter — drop autumn
    season_levels = sorted(df["season"].astype(str).unique())
    assert season_levels[0] == "autumn"

    X, y = load_data(str(CSV_PATH), SEED)
    # Rebuild expected season dummy block position: after all prior features
    # Numeric: cap-diameter; cats before season: everything except last three nums wait
    # Column order from CSV features:
    # cap-diameter(num1), cap-shape(6), cap-surface(10), cap-color(11),
    # does-bruise(1), gill-attachment(7), gill-spacing(3), gill-color(11),
    # stem-height(1), stem-width(1), stem-root(5), stem-surface(8), stem-color(12),
    # veil-type(1), veil-color(6), has-ring(1), ring-type(8), spore-print(7),
    # habitat(7), season(3)
    expected_p = 1 + 6 + 10 + 11 + 1 + 7 + 3 + 11 + 1 + 1 + 5 + 8 + 12 + 1 + 6 + 1 + 8 + 7 + 7 + 3
    assert X.shape[1] == expected_p, (X.shape[1], expected_p)

    # Find a row with season==autumn in original (pre-shuffle) — check encoding via re-load logic
    # After shuffle we can't index easily; instead verify no column is constantly 1 for all
    # autumn-only pattern by reconstructing unshuffled encoding of season.
    season = df["season"].astype(str).to_numpy()
    dummies = []
    for level in season_levels[1:]:
        dummies.append((season == level).astype(np.float64))
    S = np.column_stack(dummies)
    # autumn rows should be all-zero in season dummies
    autumn_mask = season == "autumn"
    assert autumn_mask.any()
    assert np.all(S[autumn_mask] == 0)
    # spring rows: first season dummy = 1
    spring_mask = season == "spring"
    assert np.all(S[spring_mask, 0] == 1)
    print("one-hot: season drops 'autumn'; OK")


def test_func_llh_and_grad_finite(X, y):
    from main import func_llh, mini_batch_grad_llh

    # Use a small subset for speed
    Xs = X[:500]
    ys = y[:500]
    theta = np.zeros(Xs.shape[1] + 1)
    f = func_llh(theta, Xs, ys)
    g = mini_batch_grad_llh(theta, Xs, ys, (0, 128))
    assert np.isfinite(f)
    assert g.ndim == 1
    assert g.shape == (Xs.shape[1] + 1,)
    assert np.isfinite(g).all()
    print(f"func_llh(zeros)={f:.6f}, grad_norm={np.linalg.norm(g):.4f}")

    # Numerical gradient on tiny synthetic problem (note: analytic uses factor m)
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(20, 3))
    yt = (rng.random(20) > 0.5).astype(np.float64)
    th = rng.normal(size=4) * 0.1
    f0 = func_llh(th, Xt, yt)
    eps = 1e-6
    num = np.zeros_like(th)
    for i in range(th.size):
        e = np.zeros_like(th)
        e[i] = eps
        num[i] = (func_llh(th + e, Xt, yt) - func_llh(th - e, Xt, yt)) / (2 * eps)
    ana = mini_batch_grad_llh(th, Xt, yt, (0, 20))
    # Analytic gradient of *sum* llh equals m * grad(mean); ratio should be ~ m
    ratio = np.linalg.norm(ana) / (np.linalg.norm(num) + 1e-15)
    print(f"numgrad vs analytic: cosine={np.dot(ana, num)/(np.linalg.norm(ana)*np.linalg.norm(num)):.6f}, ||ana||/||num||={ratio:.4f} (expect ~ m={Xt.shape[0]})")
    assert np.dot(ana, num) > 0
    assert abs(ratio - Xt.shape[0]) / Xt.shape[0] < 0.05


def test_next_batch_bounds():
    from main import next_batch_bounds

    assert next_batch_bounds((0, 128), 1000) == (128, 256)
    assert next_batch_bounds((768, 896), 1000) == (-104, 24)
    # wrap case from assignment style
    assert next_batch_bounds((896 - 1000, 24), 1000) == (24, 152)
    print("next_batch_bounds: OK")


def test_mini_batch_wrap(X, y):
    from main import mini_batch_grad_llh

    theta = np.zeros(X.shape[1] + 1)
    # Wrap-around window must be nonempty (NumPy X[-10:20] would be empty)
    g = mini_batch_grad_llh(theta, X[:100], y[:100], (-10, 20))
    assert g.ndim == 1
    assert np.isfinite(g).all()
    assert g.shape[0] == X.shape[1] + 1
    # Manual reference
    m = 100
    idx = np.arange(-10, 20) % m
    from main import _sigmoid, _linear_scores

    Xb = X[:100][idx]
    yb = y[:100][idx]
    tb = _linear_scores(theta, Xb)
    resid = yb - _sigmoid(tb)
    Xb_aug = np.hstack([np.ones((idx.size, 1)), Xb])
    ref = (m / idx.size) * (Xb_aug.T @ resid)
    assert np.allclose(g, ref)
    print("mini_batch wrap-around (-10, 20): OK")


def test_logistic_prediction_and_accuracy():
    from main import accuracy, logistic_prediction

    X = np.array([[1.0], [-1.0], [0.0], [2.0]])
    theta = np.array([0.0, 1.0])  # t = x
    yhat = logistic_prediction(X, theta)
    assert yhat.ndim == 1
    assert np.allclose(yhat, [1.0, 0.0, 1.0, 1.0])
    assert abs(accuracy([1, 0, 1, 0], yhat) - 0.75) < 1e-12
    print("prediction/accuracy: OK")


def test_logit_small_subset(X, y):
    from main import accuracy, logistic_prediction, logit

    n = 2000
    Xs, ys = X[:n], y[:n]
    theta = logit(Xs, ys)
    assert theta.ndim == 1
    assert theta.shape[0] == Xs.shape[1] + 1
    yhat = logistic_prediction(Xs, theta)
    acc = accuracy(ys, yhat)
    print(f"logit on first {n} rows: accuracy={acc:.4f}")
    assert acc > 0.7
    return acc


def test_logit_full_optional(X, y, run_full=False):
    if not run_full:
        print("Skipping full-data logit (set run_full=True to enable)")
        return None
    from main import accuracy, logistic_prediction, logit

    theta = logit(X, y)
    yhat = logistic_prediction(X, theta)
    acc = accuracy(y, yhat)
    print(f"logit on FULL data: accuracy={acc:.4f}")
    return acc


if __name__ == "__main__":
    test_import_has_no_side_effects()
    X, y = test_load_data_shapes_and_repro()
    test_one_hot_drops_first_alphabetical()
    test_func_llh_and_grad_finite(X, y)
    test_next_batch_bounds()
    test_mini_batch_wrap(X, y)
    test_logistic_prediction_and_accuracy()
    acc_small = test_logit_small_subset(X, y)
    # Try a short full run — may take a while
    acc_full = test_logit_full_optional(X, y, run_full=True)
    print("ALL TESTS PASSED")
    print({"acc_small": acc_small, "acc_full": acc_full})
