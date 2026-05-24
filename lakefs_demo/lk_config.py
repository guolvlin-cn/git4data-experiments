"""lakeFS connection config + the (X, y) <-> CSV codec.

Credentials come from `.lakefs.env` (gitignored). Each incoming batch is stored
as one CSV object `data/batch_NN.csv`; lexical filename order == ingestion
order, which keeps incremental-training reproduction deterministic and lets
lakeFS diff additions/modifications at object granularity.
"""
import io
import os

import numpy as np

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(ROOT, ".lakefs.env")

REPO = "ml-git4data-demo"
DATA_PREFIX = "data/"


def load_env():
    if not os.path.exists(ENV):
        raise SystemExit(
            ".lakefs.env not found — copy .lakefs.env.example and fill in OSS + "
            "lakeFS credentials first."
        )
    env = {}
    with open(ENV) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def client():
    import lakefs

    env = load_env()
    return lakefs.Client(
        host=env["LAKEFS_ENDPOINT"],
        username=env["LAKEFS_ACCESS_KEY_ID"],
        password=env["LAKEFS_SECRET_ACCESS_KEY"],
    )


def storage_namespace():
    env = load_env()
    return f"s3://{env['OSS_BUCKET']}/{REPO}"


def clean_namespace():
    """Delete every object under the repo's OSS prefix.

    lakeFS refuses to (re)create a repo whose storage namespace already holds
    lakeFS objects (the `_lakefs/dummy` marker). To keep the demo idempotent we
    wipe the prefix first, talking to OSS via its S3-compatible API with boto3.
    """
    import boto3
    from botocore.config import Config

    env = load_env()
    s3 = boto3.client(
        "s3",
        endpoint_url=env["OSS_ENDPOINT"],
        region_name=env["OSS_REGION"],
        aws_access_key_id=env["OSS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["OSS_ACCESS_KEY_SECRET"],
        # OSS requires virtual-hosted-style addressing (bucket.endpoint),
        # not the path-style boto3 defaults to for custom endpoints.
        config=Config(s3={"addressing_style": "virtual"}),
    )
    bucket = env["OSS_BUCKET"]
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    # Single DELETEs rather than batch DeleteObjects: OSS rejects the batch API
    # without a Content-MD5 header (which recent boto3 omits). Object counts per
    # run are tiny, so per-object deletes are fine.
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{REPO}/"):
        for o in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=o["Key"])
            deleted += 1
    return deleted


def batch_path(batch_id):
    return f"{DATA_PREFIX}batch_{batch_id:02d}.csv"


# ---- (ids, X, y) <-> CSV bytes ----
def encode(batch_id, X, y):
    base = batch_id * config.BATCH_SIZE
    buf = io.StringIO()
    buf.write("id," + ",".join(f"f{i}" for i in range(config.FEATURE_DIM)) + ",label\n")
    for i in range(len(y)):
        feats = ",".join(repr(float(v)) for v in X[i])
        buf.write(f"{base + i},{feats},{int(y[i])}\n")
    return buf.getvalue().encode()


def decode(raw):
    rows = [r for r in raw.decode().splitlines() if r][1:]  # drop header
    arr = np.array([[float(c) for c in r.split(",")] for r in rows], dtype=np.float64)
    ids = arr[:, 0].astype(np.int64)
    return ids, arr[:, 1:-1], arr[:, -1].astype(np.int64)
