"""git-for-data operations on MatrixOne, named after their git analogues.

| git concept      | MatrixOne primitive                                       |
|------------------|-----------------------------------------------------------|
| commit / tag     | CREATE SNAPSHOT ... FOR TABLE/DATABASE                    |
| read at a commit | SELECT ... {snapshot = 'name'}  (time travel)            |
| clone (copy)     | CREATE TABLE/DATABASE ... CLONE ... {snapshot='name'}     |
| branch           | DATA BRANCH CREATE TABLE dst FROM src   (lineage-tracked) |
| diff             | DATA BRANCH DIFF target AGAINST base    (row-level)       |
| merge            | DATA BRANCH MERGE src INTO dst WHEN CONFLICT ...          |
| cherry-pick      | DATA BRANCH PICK src INTO dst KEYS(...)                   |
| reset / revert   | RESTORE ACCOUNT .. DATABASE .. TABLE .. FROM SNAPSHOT     |

The DATA BRANCH family (verified on MatrixOne v3.0.11) gives real git
semantics: CREATE establishes a lineage DAG so DIFF/MERGE can auto-detect the
Lowest Common Ancestor and do row-level 3-way merges with conflict handling.

Identifiers (db/table/snapshot names) are interpolated, not parameterised,
because MySQL placeholders only bind values, not identifiers. All names here
are demo-controlled constants, never user input.
"""
import config


# ---- commit / tag ----------------------------------------------------------
def snapshot(mo, name, db, table=None):
    """Tag the current state. Table-level if `table` given, else database-level."""
    if table:
        mo.execute(f"CREATE SNAPSHOT {name} FOR TABLE {db} {table}")
    else:
        mo.execute(f"CREATE SNAPSHOT {name} FOR DATABASE {db}")
    return name


def drop_snapshot(mo, name):
    mo.execute(f"DROP SNAPSHOT IF EXISTS {name}")


def list_snapshots(mo):
    return mo.query("SHOW SNAPSHOTS")


# ---- read at a commit (time travel) ----------------------------------------
def count_at(mo, db, table, snapshot=None):
    if snapshot:
        return mo.scalar(f"SELECT COUNT(*) FROM {db}.{table} {{snapshot = '{snapshot}'}}")
    return mo.scalar(f"SELECT COUNT(*) FROM {db}.{table}")


def fetch_at(mo, db, table, cols, snapshot=None, order_by="id"):
    """Read rows as of a snapshot (or live if snapshot is None)."""
    col_sql = ", ".join(cols)
    snap = f" {{snapshot = '{snapshot}'}}" if snapshot else ""
    return mo.query(f"SELECT {col_sql} FROM {db}.{table}{snap} ORDER BY {order_by}")


# ---- branch ----------------------------------------------------------------
def clone_table(mo, src_db, src_table, dst_db, dst_table, snapshot=None):
    snap = f" {{snapshot = '{snapshot}'}}" if snapshot else ""
    mo.execute(
        f"CREATE TABLE {dst_db}.{dst_table} CLONE {src_db}.{src_table}{snap}"
    )


def clone_db(mo, src_db, dst_db, snapshot=None):
    snap = f" {{snapshot = '{snapshot}'}}" if snapshot else ""
    mo.execute(f"CREATE DATABASE {dst_db} CLONE {src_db}{snap}")


# ---- reset / revert --------------------------------------------------------
def restore_table(mo, db, table, snapshot, account=None):
    account = account or config.mo_account_name()
    mo.execute(
        f"RESTORE ACCOUNT {account} DATABASE {db} TABLE {table} "
        f"FROM SNAPSHOT {snapshot}"
    )


# ---- branch (lineage-tracked, enables native diff/merge) -------------------
def branch_create(mo, db, src_table, dst_table):
    """DATA BRANCH CREATE: branch a table, recording lineage for LCA detection."""
    mo.execute(
        f"DATA BRANCH CREATE TABLE {db}.{dst_table} FROM {db}.{src_table}"
    )


def _snap_hint(snapshot):
    return f" {{snapshot = '{snapshot}'}}" if snapshot else ""


# ---- diff (native, row-level) ----------------------------------------------
def branch_diff_summary(mo, db, target, base, target_snap=None, base_snap=None):
    """DATA BRANCH DIFF ... OUTPUT SUMMARY -> {'INSERTED': n, 'DELETED': n, 'UPDATED': n}.

    Counts are `target` relative to `base` (rows in target not in base ==
    INSERTED). `target`/`base` may be different tables or the same table at two
    snapshots.
    """
    rows = mo.query(
        f"DATA BRANCH DIFF {db}.{target}{_snap_hint(target_snap)} "
        f"AGAINST {db}.{base}{_snap_hint(base_snap)} OUTPUT SUMMARY"
    )
    return {r[0]: int(r[1]) for r in rows}


# ---- merge (native, 3-way, conflict-aware) ---------------------------------
def branch_merge(mo, db, src, dst, conflict="ACCEPT"):
    """DATA BRANCH MERGE src INTO dst. conflict in {FAIL, SKIP, ACCEPT}."""
    mo.execute(
        f"DATA BRANCH MERGE {db}.{src} INTO {db}.{dst} WHEN CONFLICT {conflict}"
    )


# ---- cherry-pick (native, row subset) --------------------------------------
def branch_pick(mo, db, src, dst, keys, conflict="ACCEPT"):
    """DATA BRANCH PICK: promote only the given primary keys from src to dst."""
    keylist = ", ".join(str(int(k)) for k in keys)
    mo.execute(
        f"DATA BRANCH PICK {db}.{src} INTO {db}.{dst} "
        f"KEYS({keylist}) WHEN CONFLICT {conflict}"
    )
