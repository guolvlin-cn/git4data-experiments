"""Incremental (online) classifier used across both demos.

SGDClassifier.partial_fit is the canonical sklearn primitive for continuous
learning: each incoming batch nudges the same weights rather than retraining
from scratch. Training is deterministic given the same data, order and seed —
which is exactly what lets us *reproduce* a historical model from a versioned
data snapshot.
"""
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

CLASSES = [0, 1]


class IncrementalModel:
    def __init__(self, seed=42):
        # constant LR (no iterate averaging): clean batches converge stably,
        # while a single poisoned batch still visibly moves the weights — which
        # is what makes the rollback story legible.
        self.clf = SGDClassifier(
            loss="log_loss", learning_rate="constant", eta0=0.01,
            random_state=seed,
        )
        self._started = False

    def update(self, X, y):
        """Apply one incremental step (one batch) to the running weights."""
        if not self._started:
            self.clf.partial_fit(X, y, classes=CLASSES)
            self._started = True
        else:
            self.clf.partial_fit(X, y)
        return self

    def evaluate(self, X, y):
        pred = self.clf.predict(X)
        return {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
        }


def train_from_scratch(batches, seed=42):
    """Train a fresh model over an ordered list of (X, y) batches.

    Used to reproduce a historical model: feed the exact versioned batches,
    in the same order, and you get the same weights back.
    """
    m = IncrementalModel(seed=seed)
    for X, y in batches:
        m.update(X, y)
    return m
