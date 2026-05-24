"""Experiment: SFT dataset curation done IN-PLACE on a versioned table.

SFT / instruction-tuning data is inherently tabular: (prompt, response) + a lot
of metadata (source, prompt hash, token count, quality score, language, split).
Curating it means dedup, quality/length filtering, decontamination against the
eval set, and mixing sources — all of which are SQL set-operations.

MatrixOne lets you run that whole pipeline as SQL *directly on a versioned
table*, snapshot each release, and get ROW-LEVEL provenance of what each step
dropped (DATA BRANCH DIFF). lakeFS would store the data as files and needs an
external engine (Spark/DuckDB) for every transform; its diff is file-level, so
it can't tell you "these 612 rows were dropped as duplicates".

Run:  python3 -m experiments.exp_sft_curation
"""
import random
import time

from matrixone.mo_client import MO

DB = "mld_sft"
N = 8000
MAX_TOKENS = 2048
MIN_QUALITY = 0.5


def gen_rows(n, seed=7):
    rng = random.Random(seed)
    sources = ["webA", "webB", "human"]
    langs = ["en", "en", "en", "zh", "fr"]
    eval_hashes = set(range(900000, 900050))  # the held-out eval prompts
    rows = []
    for i in range(n):
        # ~12% exact-duplicate prompts (reuse a small hash space)
        if rng.random() < 0.12:
            h = rng.randint(1, 400)          # collides -> duplicates
        elif rng.random() < 0.05:
            h = rng.choice(list(eval_hashes))  # ~5% contaminated with eval set
        else:
            h = rng.randint(10000, 99999)
        n_tok = rng.randint(20, 1500) if rng.random() > 0.10 else rng.randint(2049, 6000)
        quality = round(rng.random(), 3)
        rows.append((i, rng.choice(sources), h, n_tok, quality, rng.choice(langs)))
    return rows, sorted(eval_hashes)


def count(mo, where=""):
    return int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.sft" + (f" WHERE {where}" if where else "")))


def main():
    rows, eval_hashes = gen_rows(N)
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.sft (id BIGINT PRIMARY KEY, source VARCHAR(16), "
            f"prompt_hash BIGINT, n_tokens INT, quality DOUBLE, lang VARCHAR(8))"
        )
        mo.executemany(
            f"INSERT INTO {DB}.sft VALUES (%s,%s,%s,%s,%s,%s)", rows
        )
        mo.execute(f"CREATE TABLE {DB}.eval_set (prompt_hash BIGINT PRIMARY KEY)")
        mo.executemany(f"INSERT INTO {DB}.eval_set VALUES (%s)", [(h,) for h in eval_hashes])

        for s in ("mld_sft_raw", "mld_sft_clean"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS {s}")
        mo.execute(f"CREATE SNAPSHOT mld_sft_raw FOR TABLE {DB} sft")

        print(f"raw ingested: {count(mo)} examples (snapshot mld_sft_raw)\n")
        print("curation pipeline (all in-place SQL on the versioned table):")
        t0 = time.perf_counter()

        before = count(mo)
        mo.execute(
            f"DELETE FROM {DB}.sft WHERE id IN (SELECT id FROM (SELECT id, "
            f"ROW_NUMBER() OVER (PARTITION BY prompt_hash ORDER BY quality DESC) rn "
            f"FROM {DB}.sft) z WHERE rn > 1)"
        )
        print(f"  1. dedup by prompt_hash (keep best quality) : -{before - count(mo)}")

        before = count(mo)
        mo.execute(f"DELETE FROM {DB}.sft WHERE quality < {MIN_QUALITY}")
        print(f"  2. drop quality < {MIN_QUALITY}                      : -{before - count(mo)}")

        before = count(mo)
        mo.execute(f"DELETE FROM {DB}.sft WHERE n_tokens > {MAX_TOKENS}")
        print(f"  3. drop n_tokens > {MAX_TOKENS}                   : -{before - count(mo)}")

        before = count(mo)
        mo.execute(
            f"DELETE FROM {DB}.sft WHERE prompt_hash IN (SELECT prompt_hash FROM {DB}.eval_set)"
        )
        print(f"  4. decontaminate vs eval set               : -{before - count(mo)}")

        dt = (time.perf_counter() - t0) * 1000
        mo.execute(f"CREATE SNAPSHOT mld_sft_clean FOR TABLE {DB} sft")
        print(f"\ncurated: {count(mo)} examples (snapshot mld_sft_clean)  "
              f"[pipeline {dt:.0f} ms]")

        # ROW-LEVEL provenance of the whole curation, native:
        summary = {r[0]: int(r[1]) for r in mo.query(
            f"DATA BRANCH DIFF {DB}.sft {{snapshot='mld_sft_clean'}} "
            f"AGAINST {DB}.sft {{snapshot='mld_sft_raw'}} OUTPUT SUMMARY")}
        print(f"DATA BRANCH DIFF clean AGAINST raw -> DELETED={summary.get('DELETED',0)} "
              f"(exact row-level provenance of what curation removed)")

        # dataset composition by source/lang — an SQL group-by on the version
        print("\nfinal mix (SQL group-by on the curated version):")
        for src, lang, c in mo.query(
            f"SELECT source, lang, COUNT(*) FROM {DB}.sft GROUP BY source, lang "
            f"ORDER BY source, lang"):
            print(f"    {src:<6} {lang:<3} {c}")

        # reproducibility: both releases are pinned and queryable any time
        raw_n = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.sft {{snapshot='mld_sft_raw'}}"))
        clean_n = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.sft {{snapshot='mld_sft_clean'}}"))
        print(f"\nreproducible: raw_v0={raw_n}, clean_v1={clean_n} (both time-travelable)")

        # branch a stricter-cleaning experiment, diff row-level, decide
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.sft_strict FROM {DB}.sft")
        mo.execute(f"DELETE FROM {DB}.sft_strict WHERE quality < 0.7")
        d = {r[0]: int(r[1]) for r in mo.query(
            f"DATA BRANCH DIFF {DB}.sft_strict AGAINST {DB}.sft OUTPUT SUMMARY")}
        print(f"\nexperiment branch (quality>=0.7): DATA BRANCH DIFF shows "
              f"DELETED={d.get('DELETED',0)} more rows would be dropped — decide, then "
              f"MERGE or discard the branch.")

        for s in ("mld_sft_raw", "mld_sft_clean"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS {s}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        print("\nWhy MatrixOne fits SFT curation: dedup/filter/decontaminate/mix are SQL "
              "set-ops run IN-PLACE on a versioned table, with row-level provenance and\n"
              "reproducible releases. lakeFS would store jsonl + need Spark/DuckDB for "
              "every transform, and its file-level diff can't say which examples were cut.")


if __name__ == "__main__":
    main()
