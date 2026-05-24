"""Synthetic streaming data source for the continuous-learning demo.

Ground truth is a *stationary* linear boundary, so a single clean holdout set
stays valid across the whole run. Each incoming batch is fresh i.i.d. samples
labelled by that boundary plus a little label noise. This gives a clean
"more data -> better model" curve, against which a poisoned batch (heavy label
flipping) visibly degrades the model — motivating rollback.

Everything is seeded so any batch / holdout can be reproduced bit-for-bit,
which is what the version-control reproducibility demo relies on.
"""
import numpy as np

import config

# Fixed ground-truth weight vector + bias (seeded once, never changes).
_GT_RNG = np.random.default_rng(20240501)
TRUE_W = _GT_RNG.normal(size=config.FEATURE_DIM)
TRUE_B = 0.0


def _label(X, rng, noise):
    margin = X @ TRUE_W + TRUE_B
    y = (margin > 0).astype(np.int64)
    if noise > 0:
        flip = rng.random(len(y)) < noise
        y[flip] = 1 - y[flip]
    return y


def make_batch(batch_id, n=None, poison=False, seed=None):
    """Return (X, y) for one incoming batch.

    Deterministic in (batch_id) unless an explicit seed is given. A poisoned
    batch flips a large fraction of labels to simulate a bad upstream source.
    """
    n = n or config.BATCH_SIZE
    seed = config.GLOBAL_SEED + batch_id if seed is None else seed
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, config.FEATURE_DIM))
    noise = config.POISON_FRACTION if poison else config.LABEL_NOISE
    y = _label(X, rng, noise)
    return X, y


def make_holdout():
    """Fixed clean test set — same every run (stable ground truth)."""
    rng = np.random.default_rng(999999)
    X = rng.normal(size=(config.HOLDOUT_SIZE, config.FEATURE_DIM))
    y = _label(X, rng, noise=0.0)
    return X, y
