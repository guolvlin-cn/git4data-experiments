"""Scenario 2: multimodal (image / video / document) training set.

The raw bytes (images, video, docs) are large and unstructured — that's lakeFS's
home turf (content-addressed, byte-level versioning, multi-engine read). But the
*catalog* around them is structured: which file, modality, label, split, and an
embedding. MatrixOne can:
  - reference the files via STAGE + datalink and read them with load_file(),
  - store embeddings in a `vecf32` column and do semantic near-dup detection in SQL,
  - version the catalog (labels/splits/embeddings/refs) with git4data.

Honest boundary (verified in exp_stage_datalink): git4data versions the catalog
& references, NOT the external file bytes — overwriting a blob is invisible to a
snapshot. For byte-level versioning of the media itself, pair with lakeFS.

Requires .lakefs.env (OSS). Run:  python3 -m experiments.exp_multimodal_catalog
"""
import boto3
from botocore.config import Config

from lakefs_demo import lk_config as lk
from matrixone.mo_client import MO

PREFIX = "mld_mm_demo/"
STAGE = "mld_mm_stage"
DB = "mld_mm"
# (id, modality, filename, label, split, embedding) — items 1&2 are near-duplicates
ITEMS = [
    (1, "image", "cat_01.jpg", 1, "train", [0.10, 0.20, 0.30, 0.40]),
    (2, "image", "cat_02.jpg", 1, "train", [0.11, 0.19, 0.31, 0.39]),  # near-dup of 1
    (3, "image", "dog_01.jpg", 0, "train", [0.90, 0.80, 0.10, 0.20]),
    (4, "video", "clip_01.mp4", 0, "val", [0.50, 0.50, 0.50, 0.50]),
    (5, "doc", "report_01.txt", 1, "train", [0.20, 0.10, 0.90, 0.30]),
]


def oss(env):
    return boto3.client(
        "s3", endpoint_url=env["OSS_ENDPOINT"], region_name=env["OSS_REGION"],
        aws_access_key_id=env["OSS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["OSS_ACCESS_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"))


def wipe(s3, bucket, prefix):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=o["Key"])


def main():
    env = lk.load_env()
    bucket = env["OSS_BUCKET"]
    endpoint = env["OSS_ENDPOINT"].replace("https://", "").replace("http://", "")
    s3 = oss(env)
    wipe(s3, bucket, PREFIX)
    for _id, mod, fn, *_ in ITEMS:
        s3.put_object(Bucket=bucket, Key=PREFIX + fn,
                      Body=f"<{mod} bytes for {fn}>".encode())

    with MO() as mo:
        mo.execute(f"CREATE DATABASE IF NOT EXISTS {DB}")
        mo.execute(f"DROP STAGE IF EXISTS {STAGE}")
        mo.execute(
            f"CREATE STAGE {STAGE} URL='s3://{bucket}/{PREFIX}' CREDENTIALS={{"
            f"'AWS_KEY_ID'='{env['OSS_ACCESS_KEY_ID']}',"
            f"'AWS_SECRET_KEY'='{env['OSS_ACCESS_KEY_SECRET']}',"
            f"'AWS_REGION'='{env['OSS_REGION']}','PROVIDER'='Minio','ENDPOINT'='{endpoint}'}}")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.catalog")
        mo.execute(
            f"CREATE TABLE {DB}.catalog (id INT PRIMARY KEY, modality VARCHAR(8), "
            f"uri datalink, label INT, split VARCHAR(8), emb vecf32(4))")
        for _id, mod, fn, label, split, emb in ITEMS:
            mo.execute(
                f"INSERT INTO {DB}.catalog VALUES (%s,%s,cast(%s as datalink),%s,%s,%s)",
                (_id, mod, f"stage://{STAGE}/{fn}", label, split, str(emb)))
        print(f"catalogued {len(ITEMS)} multimodal items (image/video/doc) referencing OSS\n")

        print("read a document's bytes via load_file():")
        print("  doc#5 ->", mo.scalar(f"SELECT load_file(uri) FROM {DB}.catalog WHERE id=5"))

        print("\nsemantic near-dup detection in SQL (l2_distance on embeddings):")
        for a, b, d in mo.query(
            f"SELECT a.id, b.id, round(l2_distance(a.emb,b.emb),4) FROM {DB}.catalog a "
            f"JOIN {DB}.catalog b ON a.id<b.id AND a.modality=b.modality "
            f"ORDER BY 3 LIMIT 3"):
            flag = "  <-- near-duplicate" if d < 0.05 else ""
            print(f"  items {a},{b}: dist={d}{flag}")

        # curate on a branch: drop the near-dup, fix a split; row-level diff
        mo.execute(f"DROP TABLE IF EXISTS {DB}.catalog_v2")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.catalog_v2 FROM {DB}.catalog")
        mo.execute(f"DELETE FROM {DB}.catalog_v2 WHERE id=2")            # drop near-dup
        mo.execute(f"UPDATE {DB}.catalog_v2 SET split='test' WHERE id=4")  # fix a split
        d = {r[0]: int(r[1]) for r in mo.query(
            f"DATA BRANCH DIFF {DB}.catalog_v2 AGAINST {DB}.catalog OUTPUT SUMMARY")}
        print(f"\ncurated catalog branch: DATA BRANCH DIFF = "
              f"DELETED={d.get('DELETED',0)} (near-dup) UPDATED={d.get('UPDATED',0)} (split fix)")

        mo.execute(f"DROP TABLE IF EXISTS {DB}.catalog_v2")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.catalog")
        mo.execute(f"DROP STAGE IF EXISTS {STAGE}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
    wipe(s3, bucket, PREFIX)
    print("\nSplit of duties: MatrixOne versions the CATALOG (refs+labels+splits+embeddings)"
          " and does semantic SQL; the media BYTES are not versioned by snapshots\n"
          "(datalink = reference). Pair with lakeFS for byte-level media versioning.")


if __name__ == "__main__":
    main()
