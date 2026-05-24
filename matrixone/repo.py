"""Dataset + model-registry storage layer on MatrixOne.

The `samples` table is the growing training set (one row per labelled sample).
Row ids are assigned in generation order (batch_id * BATCH_SIZE + i) so that
reading back `ORDER BY id` reproduces the exact streaming order — essential for
deterministic model reproduction.

`model_registry` records, for each trained model version, the data snapshot it
was trained on plus its holdout metrics. That snapshot pointer is the data
lineage link that makes a model reproducible.
"""
import numpy as np

import config

FEATS = [f"f{i}" for i in range(config.FEATURE_DIM)]


def reset_db(mo):
    mo.execute(f"DROP DATABASE IF EXISTS {config.DB}")
    mo.execute(f"CREATE DATABASE {config.DB}")
    cols = ", ".join(f"{f} DOUBLE" for f in FEATS)
    mo.execute(
        f"CREATE TABLE {config.DB}.{config.SAMPLES_TABLE} ("
        f"id BIGINT PRIMARY KEY, batch INT, {cols}, label INT)"
    )
    mo.execute(
        f"CREATE TABLE {config.DB}.{config.REGISTRY_TABLE} ("
        "model_version VARCHAR(64) PRIMARY KEY, data_snapshot VARCHAR(64), "
        "n_samples BIGINT, accuracy DOUBLE, f1 DOUBLE, note VARCHAR(256), "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )


def insert_batch(mo, batch_id, X, y, db=None, table=None):
    db = db or config.DB
    table = table or config.SAMPLES_TABLE
    base = batch_id * config.BATCH_SIZE
    placeholders = ", ".join(["%s"] * (2 + config.FEATURE_DIM + 1))
    sql = (
        f"INSERT INTO {db}.{table} (id, batch, {', '.join(FEATS)}, label) "
        f"VALUES ({placeholders})"
    )
    rows = [
        (base + i, batch_id, *(float(v) for v in X[i]), int(y[i]))
        for i in range(len(y))
    ]
    mo.executemany(sql, rows)
    return len(rows)


def load_xy(mo, snapshot=None, db=None, table=None):
    """All rows (optionally as-of a snapshot) as (X, y), ordered by id."""
    db = db or config.DB
    table = table or config.SAMPLES_TABLE
    snap = f" {{snapshot = '{snapshot}'}}" if snapshot else ""
    rows = mo.query(
        f"SELECT {', '.join(FEATS)}, label FROM {db}.{table}{snap} ORDER BY id"
    )
    if not rows:
        return np.empty((0, config.FEATURE_DIM)), np.empty((0,), dtype=np.int64)
    arr = np.array(rows, dtype=np.float64)
    return arr[:, :-1], arr[:, -1].astype(np.int64)


def load_batches(mo, snapshot=None, db=None, table=None):
    """Ordered list of (X, y) per batch — for reproducing incremental training."""
    db = db or config.DB
    table = table or config.SAMPLES_TABLE
    snap = f" {{snapshot = '{snapshot}'}}" if snapshot else ""
    batch_ids = [
        r[0]
        for r in mo.query(
            f"SELECT DISTINCT batch FROM {db}.{table}{snap} ORDER BY batch"
        )
    ]
    out = []
    for b in batch_ids:
        rows = mo.query(
            f"SELECT {', '.join(FEATS)}, label FROM {db}.{table}{snap} "
            f"WHERE batch = {b} ORDER BY id"
        )
        arr = np.array(rows, dtype=np.float64)
        out.append((arr[:, :-1], arr[:, -1].astype(np.int64)))
    return out


def register_model(mo, version, snapshot, n_samples, metrics, note=""):
    mo.execute(
        f"INSERT INTO {config.DB}.{config.REGISTRY_TABLE} "
        "(model_version, data_snapshot, n_samples, accuracy, f1, note) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (version, snapshot, n_samples, metrics["accuracy"], metrics["f1"], note),
    )


def show_registry(mo):
    return mo.query(
        f"SELECT model_version, data_snapshot, n_samples, accuracy, f1, note "
        f"FROM {config.DB}.{config.REGISTRY_TABLE} ORDER BY created_at"
    )
