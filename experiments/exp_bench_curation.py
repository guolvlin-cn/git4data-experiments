"""Benchmark: same SFT curation, two architectures.

  A) MatrixOne  — curate IN-PLACE with SQL on a versioned table (data never moves),
                  then snapshot the result.
  B) lakeFS+DuckDB — data is a versioned parquet object; to curate you must READ it
                  out of lakeFS, COMPUTE in DuckDB, WRITE a new parquet, COMMIT.

The point isn't "which SQL engine is faster" (both are fast) — it's the
architecture: MatrixOne keeps compute where the data + versions live, so a
curation cycle is just the SQL; the lakeFS path pays a full read+write+commit
round-trip to object storage every cycle (and rewrites the whole dataset object,
since file versioning is whole-object).

Deterministic data (formula on row index) so both curate the identical set.
Run:  python3 -m experiments.exp_bench_curation
"""
import io
import statistics
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

import config

import lakefs
from lakefs_demo import lk_config as lk
from matrixone.mo_client import MO

SIZES = [50_000, 200_000]
DB = "mld_bench"
REPO = "ml-bench"
EVAL_LO, EVAL_HI = 900_000, 900_049
MIN_Q, MAX_TOK = 0.5, 2048


def mo_generate_sql(n):
    # deterministic raw data from row index (planted dups / contamination / long / lowq)
    return (
        f"INSERT INTO {DB}.sft SELECT result AS id, "
        f"CASE result%3 WHEN 0 THEN 'webA' WHEN 1 THEN 'webB' ELSE 'human' END, "
        f"CASE WHEN result%8=0 THEN result%400 "
        f"     WHEN result%20=0 THEN {EVAL_LO}+result%50 ELSE 10000+result END, "
        f"CASE WHEN result%10=0 THEN 3000 ELSE 20+result%1400 END, "
        f"(result%1000)/1000.0, "
        f"CASE result%5 WHEN 3 THEN 'zh' WHEN 4 THEN 'fr' ELSE 'en' END "
        f"FROM generate_series(0,{n-1}) g"
    )


def py_generate(n):
    src = ["webA", "webB", "human"]
    lang = ["en", "en", "en", "zh", "fr"]
    ids, sources, hashes, toks, quals, langs = [], [], [], [], [], []
    for i in range(n):
        ids.append(i)
        sources.append(src[i % 3])
        if i % 8 == 0:
            h = i % 400
        elif i % 20 == 0:
            h = EVAL_LO + i % 50
        else:
            h = 10000 + i
        hashes.append(h)
        toks.append(3000 if i % 10 == 0 else 20 + i % 1400)
        quals.append((i % 1000) / 1000.0)
        langs.append(lang[i % 5])
    return pa.table({"id": ids, "source": sources, "prompt_hash": hashes,
                     "n_tokens": toks, "quality": quals, "lang": langs})


def bench_matrixone(mo, n, reps=3):
    acct = config.mo_account_name()
    mo.execute(f"DROP TABLE IF EXISTS {DB}.sft")
    mo.execute(f"DROP SNAPSHOT IF EXISTS mld_bench_raw")
    mo.execute(
        f"CREATE TABLE {DB}.sft (id BIGINT PRIMARY KEY, source VARCHAR(16), "
        f"prompt_hash BIGINT, n_tokens INT, quality DOUBLE, lang VARCHAR(8))"
    )
    t = time.perf_counter()
    mo.execute(mo_generate_sql(n))
    ingest = (time.perf_counter() - t) * 1000
    mo.execute(f"CREATE SNAPSHOT mld_bench_raw FOR TABLE {DB} sft")

    def one_curate():
        mo.execute(f"RESTORE ACCOUNT {acct} DATABASE {DB} TABLE sft FROM SNAPSHOT mld_bench_raw")
        t0 = time.perf_counter()
        mo.execute(
            f"DELETE FROM {DB}.sft WHERE id IN (SELECT id FROM (SELECT id, ROW_NUMBER() "
            f"OVER (PARTITION BY prompt_hash ORDER BY quality DESC, id) rn FROM {DB}.sft) z "
            f"WHERE rn>1)"
        )
        mo.execute(f"DELETE FROM {DB}.sft WHERE quality < {MIN_Q}")
        mo.execute(f"DELETE FROM {DB}.sft WHERE n_tokens > {MAX_TOK}")
        mo.execute(f"DELETE FROM {DB}.sft WHERE prompt_hash BETWEEN {EVAL_LO} AND {EVAL_HI}")
        return (time.perf_counter() - t0) * 1000

    one_curate()  # warmup (not timed)
    times = [one_curate() for _ in range(reps)]
    rows = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.sft"))

    mo.execute(f"DROP SNAPSHOT IF EXISTS mld_bench_raw")
    mo.execute(f"DROP TABLE IF EXISTS {DB}.sft")
    return dict(ingest=ingest, curate=statistics.median(times), rows=rows)


CURATE_SQL = f"""
SELECT id, source, prompt_hash, n_tokens, quality, lang FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY prompt_hash ORDER BY quality DESC, id) rn
  FROM raw
) WHERE rn = 1 AND quality >= {MIN_Q} AND n_tokens <= {MAX_TOK}
  AND NOT (prompt_hash BETWEEN {EVAL_LO} AND {EVAL_HI})
"""


def bench_lakefs_duckdb(repo, table, n, reps=3):
    main = repo.branch("main")
    buf = io.BytesIO()
    pq.write_table(table, buf)
    raw_bytes = buf.getvalue()

    t = time.perf_counter()
    main.object("raw.parquet").upload(data=raw_bytes, mode="wb")
    main.commit(message=f"raw {n}")
    ingest = (time.perf_counter() - t) * 1000

    rows = {"n": 0}

    def one_curate(idx):
        # READ from lakeFS -> COMPUTE in DuckDB -> WRITE+COMMIT new version to lakeFS.
        # Distinct output path per run: lakeFS is content-addressed, so committing
        # identical bytes raises "no changes".
        t0 = time.perf_counter()
        with repo.ref("main").object("raw.parquet").reader(mode="rb") as r:
            downloaded = r.read()
        read_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        raw = pq.read_table(io.BytesIO(downloaded))        # noqa: F841 (used by duckdb)
        con = duckdb.connect()
        con.register("raw", raw)
        curated = con.execute(CURATE_SQL).fetch_arrow_table()
        out = io.BytesIO()
        pq.write_table(curated, out)
        curated_bytes = out.getvalue()
        compute_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        main.object(f"curated_{idx}.parquet").upload(data=curated_bytes, mode="wb")
        main.commit(message=f"curated {idx}")
        write_commit_ms = (time.perf_counter() - t0) * 1000
        rows["n"] = curated.num_rows
        return read_ms, compute_ms, write_commit_ms

    one_curate(0)  # warmup (DuckDB JIT + connection)
    samples = [one_curate(i + 1) for i in range(reps)]
    read = statistics.median(s[0] for s in samples)
    compute = statistics.median(s[1] for s in samples)
    write_commit = statistics.median(s[2] for s in samples)
    return dict(ingest=ingest, read=read, compute=compute, write_commit=write_commit,
                rows=rows["n"], curate_total=read + compute + write_commit)


def fresh_repo(clt):
    import boto3
    from botocore.config import Config
    env = lk.load_env()
    s3 = boto3.client("s3", endpoint_url=env["OSS_ENDPOINT"], region_name=env["OSS_REGION"],
                      aws_access_key_id=env["OSS_ACCESS_KEY_ID"],
                      aws_secret_access_key=env["OSS_ACCESS_KEY_SECRET"],
                      config=Config(s3={"addressing_style": "virtual"}))
    repo = lakefs.Repository(REPO, client=clt)
    try:
        repo.delete()
    except Exception:
        pass
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=env["OSS_BUCKET"], Prefix=f"{REPO}/"):
        for o in page.get("Contents", []):
            s3.delete_object(Bucket=env["OSS_BUCKET"], Key=o["Key"])
    return lakefs.Repository(REPO, client=clt).create(
        storage_namespace=f"s3://{env['OSS_BUCKET']}/{REPO}", exist_ok=True)


def main():
    clt = lk.client()
    with MO() as mo:
        mo.execute(f"CREATE DATABASE IF NOT EXISTS {DB}")
        for n in SIZES:
            a = bench_matrixone(mo, n)
            repo = fresh_repo(clt)
            b = bench_lakefs_duckdb(repo, py_generate(n), n)
            print(f"\n===== N = {n:,} rows  (MO curated {a['rows']}, lakeFS curated {b['rows']}) =====")
            print(f"  MatrixOne (in-place SQL):")
            print(f"     ingest(generate) {a['ingest']:8.0f} ms   |   CURATE (4 DELETEs) {a['curate']:8.0f} ms")
            print(f"  lakeFS + DuckDB (pull/compute/push):")
            print(f"     ingest(write+commit) {b['ingest']:6.0f} ms")
            print(f"     CURATE total {b['curate_total']:8.0f} ms = read {b['read']:.0f} + "
                  f"duckdb {b['compute']:.0f} + write+commit {b['write_commit']:.0f}")
            print(f"  -> curation cycle: MatrixOne {a['curate']:.0f} ms  vs  "
                  f"lakeFS+DuckDB {b['curate_total']:.0f} ms "
                  f"({b['curate_total']/max(a['curate'],1):.1f}x)")
        repo = lakefs.Repository(REPO, client=clt)
        try:
            repo.delete()
        except Exception:
            pass
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
    print("\nHonest reading: at these sizes lakeFS+DuckDB is FASTER on raw curation "
          "throughput — DuckDB is a world-class in-process OLAP engine, the parquet is "
          "small, and the OSS round-trip is cheap; the remote MatrixOne pays network "
          "latency on each of the 4 sequential DELETEs. So MatrixOne's edge is NOT raw "
          "curation speed. It is: one system for store+version+compute+serve (no separate "
          "engine/orchestration), ROW-LEVEL diff/merge/cherry-pick & conflict handling "
          "(lakeFS versions whole files), in-place copy-on-write snapshots, and SQL on any "
          "version. Caveat: a co-located MatrixOne would cut its numbers a lot; and the "
          "lakeFS path rewrites the whole dataset object each cycle, so at much larger "
          "scale / many cycles its full read+write cost grows (not measured here).")


if __name__ == "__main__":
    main()
