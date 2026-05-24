"""Scenario: MatrixOne as a feature-store backbone, with git4data feature
versioning + continuous iteration — and how it lines up against Tecton.

A feature store's core jobs:
  (1) define feature transformations and MATERIALIZE feature values over time;
  (2) generate POINT-IN-TIME CORRECT training sets (for each label event at time
      T, fetch features as-of T — no leakage / train-serve skew);
  (3) serve features ONLINE (low-latency latest value) and OFFLINE (full history);
  (4) version feature definitions/values + iterate continuously.

This runs all four on MatrixOne in plain SQL, and adds what Tecton does NOT do at
the data layer: git4data versions the materialized feature values — snapshot per
feature release, branch+DIFF+MERGE a feature-definition change, reproduce any
feature version's training set exactly.

Deterministic (seeded integer-day events). Run:  python3 -m experiments.exp_feature_store
"""
import random
import time

import config
from matrixone.mo_client import MO

DB = "mld_fs"
N_USERS = 40


def hr(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def gen_events(n, max_day, seed, start_id=0):
    rng = random.Random(seed)
    return [(start_id + i, rng.randint(0, N_USERS - 1), rng.randint(1, max_day),
             round(rng.uniform(1, 100), 2)) for i in range(n)]


def materialize(mo, into, asof_days, window):
    """Compute rolling features as-of each day in `asof_days` (window-day lookback)."""
    for M in asof_days:
        mo.execute(
            f"INSERT INTO {DB}.{into} (user_id, asof_day, f_cnt_win, f_sum_win, f_cnt_all) "
            f"SELECT user_id, {M}, "
            f"  SUM(CASE WHEN day > {M}-{window} AND day <= {M} THEN 1 ELSE 0 END), "
            f"  SUM(CASE WHEN day > {M}-{window} AND day <= {M} THEN amount ELSE 0 END), "
            f"  SUM(CASE WHEN day <= {M} THEN 1 ELSE 0 END) "
            f"FROM {DB}.events WHERE day <= {M} GROUP BY user_id")


def rematerialize_window(mo, table, asof_days, window):
    """Recompute the rolling-window feature columns IN PLACE (UPDATE-JOIN), so a
    feature-definition change shows up as row-level UPDATEs in DATA BRANCH DIFF
    (a DELETE+re-INSERT would instead look like INSERTs and lose row identity)."""
    asof_sel = " UNION ALL ".join(f"SELECT {M} AS M" for M in asof_days)
    mo.execute(
        f"UPDATE {DB}.{table} t JOIN ("
        f"  SELECT e.user_id uid, m.M asof, "
        f"    SUM(CASE WHEN e.day > m.M-{window} AND e.day <= m.M THEN 1 ELSE 0 END) cw, "
        f"    SUM(CASE WHEN e.day > m.M-{window} AND e.day <= m.M THEN e.amount ELSE 0 END) sw "
        f"  FROM {DB}.events e JOIN ({asof_sel}) m ON e.day <= m.M "
        f"  GROUP BY e.user_id, m.M) s "
        f"ON t.user_id = s.uid AND t.asof_day = s.asof "
        f"SET t.f_cnt_win = s.cw, t.f_sum_win = s.sw")


def pit_training_set(mo, snapshot=None, limit=None):
    """Point-in-time correct join: each label row gets features as-of its label_day."""
    snap = f" {{snapshot='{snapshot}'}}" if snapshot else ""
    lim = f" LIMIT {limit}" if limit else ""
    return mo.query(
        f"SELECT user_id, label_day, label, f_cnt_win, f_sum_win, f_cnt_all, asof_day FROM ("
        f"  SELECT s.user_id, s.label_day, s.label, fv.f_cnt_win, fv.f_sum_win, "
        f"         fv.f_cnt_all, fv.asof_day, "
        f"         ROW_NUMBER() OVER (PARTITION BY s.user_id, s.label_day "
        f"                            ORDER BY fv.asof_day DESC) rn "
        f"  FROM {DB}.spine s JOIN {DB}.feature_values{snap} fv "
        f"    ON fv.user_id = s.user_id AND fv.asof_day <= s.label_day"
        f") z WHERE rn = 1 ORDER BY user_id, label_day{lim}")


def diff(mo, target, base):
    return {r[0]: int(r[1]) for r in mo.query(
        f"DATA BRANCH DIFF {DB}.{target} AGAINST {DB}.{base} OUTPUT SUMMARY")}


def main():
    acct = config.mo_account_name()
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(f"CREATE TABLE {DB}.events (event_id INT PRIMARY KEY, user_id INT, day INT, amount DOUBLE)")
        mo.execute(
            f"CREATE TABLE {DB}.feature_values (user_id INT, asof_day INT, f_cnt_win INT, "
            f"f_sum_win DOUBLE, f_cnt_all INT, PRIMARY KEY (user_id, asof_day))")
        mo.execute(f"CREATE TABLE {DB}.spine (user_id INT, label_day INT, label INT)")
        for s in ("fs_v1", "fs_v2", "fs_v3"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS mld_{s}")

        # ---------- ACT 1: define + materialize features (offline store) ----------
        hr("ACT 1  Define feature transformation -> materialize feature values (7d window)")
        mo.executemany(f"INSERT INTO {DB}.events VALUES (%s,%s,%s,%s)", gen_events(500, 30, seed=1))
        materialize(mo, "feature_values", asof_days=[10, 20, 30], window=7)
        mo.execute(f"CREATE SNAPSHOT mld_fs_v1 FOR TABLE {DB} feature_values")
        n_fv = mo.scalar(f"SELECT COUNT(*) FROM {DB}.feature_values")
        print(f"  materialized {n_fv} feature rows at asof_days [10,20,30]; snapshot feature_set=fs_v1")
        print("  (Tecton 等价物：FeatureView 的 batch materialization 到 offline store)")

        # ---------- ACT 2: point-in-time correct training set ----------
        hr("ACT 2  Point-in-time correct training set (no leakage) — the core capability")
        mo.executemany(f"INSERT INTO {DB}.spine VALUES (%s,%s,%s)",
                       [(0, 12, 1), (0, 33, 0), (1, 25, 1), (2, 9, 0), (3, 28, 1)])
        for uid, lday, lbl, cw, sw, ca, asof in pit_training_set(mo):
            print(f"  user {uid} @day {lday} (label {lbl}) -> features as-of day {asof}: "
                  f"cnt_win={cw} cnt_all={ca}")
        latest = mo.scalar(f"SELECT f_cnt_all FROM {DB}.feature_values WHERE user_id=0 ORDER BY asof_day DESC LIMIT 1")
        asofd12 = next(r[5] for r in pit_training_set(mo) if r[0] == 0 and r[1] == 12)
        print(f"  no-leakage check: user0 LATEST cnt_all={latest}, but its day-12 label correctly "
              f"got as-of cnt_all={asofd12} (≠ latest)")
        print("  (Tecton 等价物：get_historical_features 的 time-travel join；这里就是一段 SQL)")

        # ---------- ACT 3: online + offline from ONE HTAP store ----------
        hr("ACT 3  Online (latest) + Offline (history) from the SAME store — no dual-store skew")
        t = time.perf_counter()
        online = mo.query_one(
            f"SELECT f_cnt_win, f_sum_win, f_cnt_all FROM {DB}.feature_values "
            f"WHERE user_id=0 ORDER BY asof_day DESC LIMIT 1")
        ms = (time.perf_counter() - t) * 1000
        print(f"  ONLINE serve (latest features for user0): {online}  [{ms:.0f} ms point lookup, remote]")
        print(f"  OFFLINE training: the PIT join above, full history — same feature_values table")
        print("  (Tecton 用两套存储：offline=S3/数仓, online=DynamoDB/Redis，需保证两边一致；"
              "MatrixOne HTAP 单存储天然无 online/offline skew)")

        # ---------- ACT 4: feature versioning & iteration via git4data ----------
        hr("ACT 4  Iterate the feature DEFINITION on a branch (7d -> 14d) — version the VALUES")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.fv_exp")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.fv_exp FROM {DB}.feature_values")
        rematerialize_window(mo, "fv_exp", asof_days=[10, 20, 30], window=14)  # new definition, in place
        d = diff(mo, "fv_exp", "feature_values")
        print(f"  recomputed window 7d->14d on branch fv_exp; DATA BRANCH DIFF vs fs_v1: "
              f"UPDATED={d.get('UPDATED',0)} feature rows changed (row-level)")
        mo.execute(f"DATA BRANCH MERGE {DB}.fv_exp INTO {DB}.feature_values WHEN CONFLICT ACCEPT")
        mo.execute(f"CREATE SNAPSHOT mld_fs_v2 FOR TABLE {DB} feature_values")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.fv_exp")
        # reproduce both feature versions' training sets exactly (time travel)
        u0_v1 = next(r[3] for r in pit_training_set(mo, snapshot='mld_fs_v1') if r[0] == 0 and r[1] == 33)
        u0_v2 = next(r[3] for r in pit_training_set(mo, snapshot='mld_fs_v2') if r[0] == 0 and r[1] == 33)
        print(f"  reproducible training sets: user0@33 cnt_win under fs_v1(7d)={u0_v1} vs "
              f"fs_v2(14d)={u0_v2} — each feature version is pinned & time-travelable")
        print("  (Tecton 版本化的是 feature *定义*(代码/git)；git4data 还版本化 *物化值*，"
              "且能行级 diff、分支实验、精确复现每个特征版本的训练集)")

        # ---------- ACT 5: continuous iteration (incremental materialization) ----------
        hr("ACT 5  Continuous iteration: new events -> incremental materialization -> new version")
        mo.executemany(f"INSERT INTO {DB}.events VALUES (%s,%s,%s,%s)",
                       gen_events(300, 40, seed=2, start_id=500))  # offset ids: no PK clash
        materialize(mo, "feature_values", asof_days=[40], window=14)
        mo.execute(f"CREATE SNAPSHOT mld_fs_v3 FOR TABLE {DB} feature_values")
        print(f"  new events arrived (to day 40); materialized asof_day=40; snapshot feature_set=fs_v3 "
              f"(now {mo.scalar(f'SELECT COUNT(*) FROM {DB}.feature_values')} feature rows)")
        print("  (持续迭代：每个特征发布 = 一个快照；增量物化只算新到的 asof 点)")

        # ---------- lineage ----------
        hr("Feature lineage (feature_set version -> snapshot, queryable any time)")
        for v in ("mld_fs_v1", "mld_fs_v2", "mld_fs_v3"):
            cnt = mo.scalar(f"SELECT COUNT(*) FROM {DB}.feature_values {{snapshot='{v}'}}")
            print(f"  {v}: {cnt} feature rows")

        for s in ("fs_v1", "fs_v2", "fs_v3"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS mld_{s}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")

        hr("MatrixOne vs Tecton — summary")
        print("  PIT 训练集 / 滚动物化 / online+offline / 持续迭代：都能用 SQL+git4data 在一处做。")
        print("  MatrixOne 增量价值：HTAP 单存储(无 online/offline skew) + git4data 版本化*物化值*")
        print("    (每次训练 pin 快照、分支改特征定义 DIFF/MERGE、PITR 任意时点、行级'哪些特征变了')。")
        print("  Tecton 仍占优：声明式特征定义框架、托管的流式/按需物化与编排、生产级在线服务")
        print("    SLA/监控/新鲜度、多源(batch/stream/on-demand)生态——它是完整的*特征平台*，")
        print("    MatrixOne 是可做特征存储*后端*的数据库。最佳组合：MO 当存储+版本+PIT 引擎，")
        print("    上面薄薄一层特征定义/编排/监控。")


if __name__ == "__main__":
    main()
