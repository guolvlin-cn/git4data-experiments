"""Shared configuration for the ML continuous-learning demo.

Connection credentials are read from `.mo.cnf` (gitignored) so secrets never
live in source. The MatrixOne user string has the form `account:user:role`;
the leading `account` segment is needed for RESTORE statements.
"""
import configparser
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MO_CNF = os.path.join(ROOT, ".mo.cnf")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def mo_conn_params():
    cp = configparser.ConfigParser()
    cp.read(MO_CNF)
    c = cp["client"]
    return {
        "host": c["host"],
        "port": int(c.get("port", "6001")),
        "user": c["user"],
        "password": c["password"],
    }


def mo_account_name():
    """The tenant/account segment of the user string (before the first colon)."""
    return mo_conn_params()["user"].split(":", 1)[0]


# ---- demo domain constants ----
DB = "ml_git4data_demo"
SAMPLES_TABLE = "samples"          # the growing training dataset
REGISTRY_TABLE = "model_registry"  # model_version -> data snapshot + metrics

FEATURE_DIM = 8
HOLDOUT_SIZE = 2000        # fixed clean test set, stable ground truth
BATCH_SIZE = 1000          # samples per incoming batch
N_BATCHES = 6              # normal batches before the special scenarios
LABEL_NOISE = 0.05         # fraction of randomly flipped labels in clean data
POISON_FRACTION = 0.45     # label-flip fraction in a poisoned batch
GLOBAL_SEED = 42
