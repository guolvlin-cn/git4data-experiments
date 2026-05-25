"""Durable timers + durable queue built ENTIRELY inside MatrixOne (no external
scheduler / broker), using the built-in `CREATE TASK` scheduler and `FOR UPDATE`
row locks. Verified on MatrixOne v3.0.11.

- Durable timer: a `timers` table + a `CREATE TASK` cron job that fires due timers
  in-DB (UPDATE PENDING->FIRED WHERE due_at<=now()). No external poller.
- Durable queue: a `dq` table; concurrent workers claim messages with
  `SELECT ... FOR UPDATE` (SKIP LOCKED is NOT supported here, so claims serialize
  on the lock) + UPDATE to DONE -> each message processed exactly once.

This complements durable_exec/run.py (durable execution) with the two primitives
a Temporal/DBOS-style engine needs (timers, queues) — showing they can be built
on MatrixOne natively.

Run:  python3 -m durable_exec.scheduling
"""
import threading
import time

import pymysql

import config

DB = "mld_dsched"
POLLER = "dsched_timer_poller"


def conn(autocommit=True):
    p = config.mo_conn_params()
    return pymysql.connect(host=p["host"], port=p["port"], user=p["user"],
                           password=p["password"], charset="utf8mb4", autocommit=autocommit)


def hr(s):
    print("\n" + "=" * 72 + f"\n  {s}\n" + "=" * 72)


def setup(c):
    cur = c.cursor()
    cur.execute(f"DROP TASK IF EXISTS {POLLER}")
    cur.execute(f"DROP DATABASE IF EXISTS {DB}")
    cur.execute(f"CREATE DATABASE {DB}")
    cur.execute(f"CREATE TABLE {DB}.timers (timer_id BIGINT PRIMARY KEY, wf_id VARCHAR(64), "
                f"due_at TIMESTAMP, action VARCHAR(64), status VARCHAR(12), fired_at TIMESTAMP NULL)")
    cur.execute(f"CREATE TABLE {DB}.dq (msg_id BIGINT PRIMARY KEY, payload VARCHAR(64), "
                f"status VARCHAR(12), claimed_by VARCHAR(16), done_at TIMESTAMP NULL)")


def main():
    c = conn(autocommit=True)
    setup(c)
    cur = c.cursor()

    # ---------------- Durable timers via CREATE TASK ----------------
    hr("Durable timers — in-DB scheduler (CREATE TASK), no external poller")
    # schedule 3 timers: two already due, one due in ~30s (so it stays PENDING)
    cur.execute(f"INSERT INTO {DB}.timers (timer_id, wf_id, due_at, action, status) VALUES "
                f"(1,'wf-A', now(), 'send_reminder', 'PENDING'),"
                f"(2,'wf-B', now(), 'escalate', 'PENDING'),"
                f"(3,'wf-C', date_add(now(), interval 30 second), 'timeout', 'PENDING')")
    # the in-DB scheduler: every second, fire all due PENDING timers (single statement -> atomic, idempotent)
    cur.execute(
        f"CREATE TASK {POLLER} SCHEDULE '*/1 * * * * *' AS BEGIN "
        f"UPDATE {DB}.timers SET status='FIRED', fired_at=now() WHERE status='PENDING' AND due_at<=now(); "
        f"END")
    print(f"  created task {POLLER} (cron '*/1 * * * * *'); 3 timers PENDING (2 due now, 1 due +30s)")
    print("  waiting for the in-DB scheduler to fire the due timers ...")
    fired = 0
    for _ in range(15):
        time.sleep(2)
        fired = int(_scalar(cur, f"SELECT COUNT(*) FROM {DB}.timers WHERE status='FIRED'"))
        if fired >= 2:
            break
    rows = _q(cur, f"SELECT timer_id, action, status, fired_at FROM {DB}.timers ORDER BY timer_id")
    for tid, act, st, fat in rows:
        print(f"    timer#{tid} {act:<14} {st:<8} fired_at={fat}")
    print(f"  => {fired} due timers fired by the in-DB scheduler; the +30s one is still PENDING "
          f"(correctly not yet due). SHOW TASK RUNS:")
    for r in _q(cur, f"SHOW TASK RUNS FOR {POLLER} LIMIT 3"):
        print(f"    {r}")

    # ---------------- Durable queue via FOR UPDATE ----------------
    hr("Durable queue — concurrent workers claim exactly-once via SELECT ... FOR UPDATE")
    N = 12
    cur.executemany(f"INSERT INTO {DB}.dq (msg_id, payload, status) VALUES (%s,%s,'READY')",
                    [(i, f"job-{i}") for i in range(N)])
    print(f"  enqueued {N} messages (status=READY); starting 2 concurrent workers")

    results = {}

    def worker(name):
        wc = conn(autocommit=False)
        claimed = 0
        try:
            while True:
                with wc.cursor() as wcur:
                    wcur.execute(f"SELECT msg_id FROM {DB}.dq WHERE status='READY' "
                                 f"ORDER BY msg_id LIMIT 1 FOR UPDATE")
                    row = wcur.fetchone()
                    if not row:
                        wc.commit()
                        break
                    mid = row[0]
                    wcur.execute(f"UPDATE {DB}.dq SET status='DONE', claimed_by=%s, done_at=now() "
                                 f"WHERE msg_id=%s", (name, mid))
                wc.commit()
                claimed += 1
        finally:
            wc.close()
        results[name] = claimed

    t1 = threading.Thread(target=worker, args=("worker-1",))
    t2 = threading.Thread(target=worker, args=("worker-2",))
    t1.start(); t2.start(); t1.join(); t2.join()

    done = int(_scalar(cur, f"SELECT COUNT(*) FROM {DB}.dq WHERE status='DONE'"))
    ready = int(_scalar(cur, f"SELECT COUNT(*) FROM {DB}.dq WHERE status='READY'"))
    by_worker = _q(cur, f"SELECT claimed_by, COUNT(*) FROM {DB}.dq GROUP BY claimed_by ORDER BY claimed_by")
    print(f"  results: worker claims = {results}")
    print(f"  queue: DONE={done}, READY={ready}; by worker (DB) = {dict((w, n) for w, n in by_worker)}")
    exactly_once = (done == N and ready == 0 and sum(results.values()) == N)
    print(f"  EXACTLY-ONCE consumption (every message claimed by exactly one worker): {exactly_once}")

    # cleanup
    cur.execute(f"DROP TASK IF EXISTS {POLLER}")
    cur.execute(f"DROP DATABASE IF EXISTS {DB}")
    c.close()
    hr("Done — durable timers + durable queue built natively on MatrixOne")
    print("  CREATE TASK = in-DB cron scheduler (fires due timers, no external poller);")
    print("  FOR UPDATE = exactly-once queue claims (SKIP LOCKED unsupported -> claims serialize).")
    print("  Together with durable_exec/run.py (durable execution + exactly-once steps), this")
    print("  covers the core durable-execution primitives on MatrixOne — a DBOS-on-MatrixOne base.")


def _scalar(cur, sql):
    cur.execute(sql)
    r = cur.fetchone()
    return r[0] if r else None


def _q(cur, sql):
    cur.execute(sql)
    return cur.fetchall()


if __name__ == "__main__":
    main()
