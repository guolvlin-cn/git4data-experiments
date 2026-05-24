"""Experiment: how far can MatrixOne manage UNSTRUCTURED files, and what does
git4data actually version?

MatrixOne can reference external files via a STAGE (named S3/OSS/fs location)
and a `datalink` column, and read their bytes with `load_file()`. So a table can
catalog images/documents living in OSS, and snapshots/branches version that
catalog (which files belong to which dataset version, their labels, splits).

BUT git4data versions the REFERENCE, not the bytes: if the external blob is
overwritten, a time-travel read returns the NEW bytes. For true content
versioning you either store bytes in-table (BLOB column -> versioned) or use a
content-addressed store like lakeFS. This experiment demonstrates both points.

Also shows DATA BRANCH DIFF ... OUTPUT FILE exporting a replayable SQL patch to
the stage.

Requires .lakefs.env (OSS creds). Run:  python3 -m experiments.exp_stage_datalink
"""
import boto3
from botocore.config import Config

from lakefs_demo import lk_config as lk
from matrixone.mo_client import MO

PREFIX = "mld_stage_demo/"
STAGE = "mld_demo_stage"
DB = "mld_stage_exp"


def oss_client(env):
    return boto3.client(
        "s3", endpoint_url=env["OSS_ENDPOINT"], region_name=env["OSS_REGION"],
        aws_access_key_id=env["OSS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["OSS_ACCESS_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"),
    )


def put(s3, bucket, key, body):
    s3.put_object(Bucket=bucket, Key=key, Body=body)


def wipe_prefix(s3, bucket, prefix):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=o["Key"])


def main():
    env = lk.load_env()
    bucket = env["OSS_BUCKET"]
    endpoint = env["OSS_ENDPOINT"].replace("https://", "").replace("http://", "")
    s3 = oss_client(env)

    wipe_prefix(s3, bucket, PREFIX)
    put(s3, bucket, PREFIX + "doc1.txt", b"hello from doc1 (v1)")
    put(s3, bucket, PREFIX + "doc2.txt", b"second document content")
    put(s3, bucket, PREFIX + "diffout/.keep", b"")  # OUTPUT FILE needs an existing dir

    with MO() as mo:
        mo.execute(f"CREATE DATABASE IF NOT EXISTS {DB}")
        mo.execute(f"DROP STAGE IF EXISTS {STAGE}")
        mo.execute(
            f"CREATE STAGE {STAGE} URL='s3://{bucket}/{PREFIX}' CREDENTIALS={{"
            f"'AWS_KEY_ID'='{env['OSS_ACCESS_KEY_ID']}',"
            f"'AWS_SECRET_KEY'='{env['OSS_ACCESS_KEY_SECRET']}',"
            f"'AWS_REGION'='{env['OSS_REGION']}','PROVIDER'='Minio',"
            f"'ENDPOINT'='{endpoint}'}}"
        )

        print("== 1. Catalog unstructured OSS files via datalink, read with load_file ==")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.docs")
        mo.execute(f"CREATE TABLE {DB}.docs (id INT PRIMARY KEY, fname VARCHAR(64), ref datalink)")
        mo.execute(
            f"INSERT INTO {DB}.docs VALUES "
            f"(1,'doc1.txt', cast('stage://{STAGE}/doc1.txt' as datalink)),"
            f"(2,'doc2.txt', cast('stage://{STAGE}/doc2.txt' as datalink))"
        )
        for i in (1, 2):
            print(f"  load_file(doc{i}) -> {mo.scalar(f'SELECT load_file(ref) FROM {DB}.docs WHERE id={i}')!r}")

        print("\n== 2. Snapshot versions the REFERENCE, not the bytes ==")
        mo.execute("DROP SNAPSHOT IF EXISTS mld_docs_v1")
        mo.execute(f"CREATE SNAPSHOT mld_docs_v1 FOR TABLE {DB} docs")
        print("  snapshot mld_docs_v1 taken (doc1 == 'hello from doc1 (v1)')")
        put(s3, bucket, PREFIX + "doc1.txt", b"CHANGED doc1 (v2!!)")
        print("  >> overwrote OSS doc1.txt to v2 <<")
        live = mo.scalar(f"SELECT load_file(ref) FROM {DB}.docs WHERE id=1")
        atv1 = mo.scalar(f"SELECT load_file(ref) FROM {DB}.docs {{snapshot='mld_docs_v1'}} WHERE id=1")
        print(f"  live read      -> {live!r}")
        print(f"  time-travel@v1 -> {atv1!r}")
        print("  => git4data versioned the REFERENCE; external bytes are NOT versioned."
              if live == atv1 else "  => bytes were versioned (unexpected)")

        print("\n== 3. DATA BRANCH DIFF ... OUTPUT FILE -> replayable SQL patch on OSS ==")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.o1")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.o2")
        mo.execute(f"CREATE TABLE {DB}.o1 (id INT PRIMARY KEY, v INT)")
        mo.execute(f"INSERT INTO {DB}.o1 VALUES (1,1),(2,2),(3,3)")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.o2 FROM {DB}.o1")
        mo.execute(f"INSERT INTO {DB}.o2 VALUES (4,4)")
        mo.execute(f"UPDATE {DB}.o2 SET v=99 WHERE id=2")
        mo.execute(f"DELETE FROM {DB}.o2 WHERE id=3")
        mo.execute(f"DATA BRANCH DIFF {DB}.o2 AGAINST {DB}.o1 OUTPUT FILE 'stage://{STAGE}/diffout/'")
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX + "diffout/"):
            for o in page.get("Contents", []):
                if o["Key"].endswith(".sql"):
                    body = s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read().decode()
                    print(f"  wrote {o['Key']} ({o['Size']} B); changeset rows:")
                    for line in body.splitlines():
                        s = line.strip()
                        if " values (" in s.lower():  # del-keys + ins-rows
                            print("   ", s)

        # cleanup
        mo.execute("DROP SNAPSHOT IF EXISTS mld_docs_v1")
        mo.execute(f"DROP STAGE IF EXISTS {STAGE}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
    wipe_prefix(s3, bucket, PREFIX)
    print("\n(cleaned up stage, db, snapshot, and OSS prefix)")


if __name__ == "__main__":
    main()
