"""Scenario 4: real-time stream (Kafka -> training set) versioning.

A stream lands events continuously. lakeFS versions by *committing* — you'd batch
(commit every N seconds/events); you cannot cheaply make "one version per event",
and to read "the training set as of 14:03:07.21" you need a commit at that moment.

MatrixOne is a database: every INSERT is its own durably-committed transaction,
and PITR lets you reconstruct the table as of ANY timestamp in the retention
window — i.e. effectively "one recoverable version per transaction", with no
explicit commit step in the stream path.

Here we ingest events one-per-transaction (autocommit), capture two mid-stream
instants, then rebuild the training set as of each via PITR.

Run:  python3 -m experiments.exp_stream_versioning
"""
import time

import config
from matrixone.mo_client import MO

DB = "mld_stream"
T = "events"
N = 500          # events, each its own transaction
MARKS = [200, 380]


def cnt(mo):
    return int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.{T}"))


def main():
    acct = config.mo_account_name()
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.{T} (seq BIGINT PRIMARY KEY, payload INT, ts TIMESTAMP)"
        )
        mo.execute(f"DROP PITR IF EXISTS mld_stream_pitr")
        mo.execute(f"CREATE PITR mld_stream_pitr FOR DATABASE {DB} RANGE 1 'h'")

        marks = {}
        t0 = time.perf_counter()
        for i in range(N):
            mo.execute(
                f"INSERT INTO {DB}.{T} (seq,payload,ts) VALUES (%s,%s,now(6))",
                (i, i * 7 % 100),
            )
            if i in MARKS:
                marks[i] = mo.scalar(f"SELECT now(6)")
        elapsed = time.perf_counter() - t0
        print(f"ingested {N} events as {N} separate transactions in {elapsed:.1f}s "
              f"(~{N/elapsed:.0f} txn/s, remote cloud instance — co-located/batched "
              f"would be far higher; point is per-txn durability + PITR)")
        print(f"final table: {cnt(mo)} rows\n")

        # rebuild the training set as of each captured instant, via PITR (non-destructive)
        mo.execute(f"DROP SNAPSHOT IF EXISTS mld_stream_cur")
        mo.execute(f"CREATE SNAPSHOT mld_stream_cur FOR TABLE {DB} {T}")
        for i in MARKS:
            ts = marks[i]
            mo.execute(f'RESTORE DATABASE {DB} TABLE {T} FROM PITR mld_stream_pitr "{ts}"')
            got = cnt(mo)
            mo.execute(f"RESTORE ACCOUNT {acct} DATABASE {DB} TABLE {T} FROM SNAPSHOT mld_stream_cur")
            print(f"  PITR rebuild as of just-after-event-{i}  ('{ts}') -> {got} rows "
                  f"(expected ~{i+1})")

        print("\n=> Each transaction is a recoverable version; the training set can be "
              "reconstructed as of ANY microsecond of the stream — no batch-commit step.\n"
              "   lakeFS would need an explicit commit per desired version; per-event "
              "commits aren't practical on a high-rate stream.")

        mo.execute(f"DROP SNAPSHOT IF EXISTS mld_stream_cur")
        mo.execute(f"DROP PITR IF EXISTS mld_stream_pitr")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")


if __name__ == "__main__":
    main()
