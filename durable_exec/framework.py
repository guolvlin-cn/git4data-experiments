"""A minimal DBOS-style durable-execution framework on MatrixOne — assembling the
three primitives we verified separately into one cohesive engine:

  @workflow      register a durable, crash-resumable workflow (run by a worker)
  @transaction   a step whose business writes + checkpoint commit in ONE txn
                 (exactly-once)
  @step          a non-transactional step (external side effect; checkpointed after)
  dbos.sleep()   a DURABLE timer: records a timer row, the in-DB CREATE TASK poller
                 fires it, the workflow resumes — survives crashes
  enqueue()/worker()  a DURABLE queue: workers claim enqueued workflows via
                 SELECT ... FOR UPDATE (exactly-once dispatch), run them durably;
                 a crashed run is re-queued and resumed (completed steps skipped)

State lives in MatrixOne (wf / wf_step / dq / timers). No external scheduler/broker.
"""
import json
import threading
import time
from functools import wraps

import pymysql

import config

DB = "mld_dbos"
POLLER = "dbos_timer_poller"

_WORKFLOWS = {}            # name -> function
_ctx = threading.local()  # per-worker: conn, wf_id, attempt


def _conn(autocommit=False):
    p = config.mo_conn_params()
    return pymysql.connect(host=p["host"], port=p["port"], user=p["user"],
                           password=p["password"], charset="utf8mb4", autocommit=autocommit)


# ---------------- schema / lifecycle ----------------
def setup():
    c = _conn(autocommit=True)
    cur = c.cursor()
    cur.execute(f"DROP TASK IF EXISTS {POLLER}")
    cur.execute(f"DROP DATABASE IF EXISTS {DB}")
    cur.execute(f"CREATE DATABASE {DB}")
    cur.execute(f"CREATE TABLE {DB}.wf (wf_id VARCHAR(64) PRIMARY KEY, name VARCHAR(64), "
                f"status VARCHAR(12), input JSON, output JSON)")
    cur.execute(f"CREATE TABLE {DB}.wf_step (wf_id VARCHAR(64), step VARCHAR(64), status VARCHAR(12), "
                f"output JSON, PRIMARY KEY (wf_id, step))")
    cur.execute(f"CREATE TABLE {DB}.dq (id BIGINT PRIMARY KEY AUTO_INCREMENT, wf_name VARCHAR(64), "
                f"wf_id VARCHAR(64), input JSON, status VARCHAR(12), claimed_by VARCHAR(24), attempts INT DEFAULT 0)")
    cur.execute(f"CREATE TABLE {DB}.timers (wf_id VARCHAR(64), step VARCHAR(64), due_at TIMESTAMP, "
                f"status VARCHAR(12), PRIMARY KEY (wf_id, step))")
    # business tables
    cur.execute(f"CREATE TABLE {DB}.inventory (sku VARCHAR(16) PRIMARY KEY, qty INT)")
    cur.execute(f"CREATE TABLE {DB}.payments (wf_id VARCHAR(64) PRIMARY KEY, amount DOUBLE)")
    cur.execute(f"CREATE TABLE {DB}.emails (id BIGINT PRIMARY KEY AUTO_INCREMENT, wf_id VARCHAR(64), body VARCHAR(128))")
    cur.execute(f"INSERT INTO {DB}.inventory VALUES ('WIDGET', 100)")
    # in-DB scheduler that fires due durable timers (no external poller)
    cur.execute(f"CREATE TASK {POLLER} SCHEDULE '*/1 * * * * *' AS BEGIN "
                f"UPDATE {DB}.timers SET status='FIRED' WHERE status='PENDING' AND due_at<=now(); END")
    c.close()


def teardown():
    c = _conn(autocommit=True)
    cur = c.cursor()
    cur.execute(f"DROP TASK IF EXISTS {POLLER}")
    cur.execute(f"DROP DATABASE IF EXISTS {DB}")
    c.close()


# ---------------- decorators ----------------
def workflow(name):
    def deco(fn):
        _WORKFLOWS[name] = fn
        fn._wf_name = name
        return fn
    return deco


def _step_done(conn, wf_id, step):
    with conn.cursor() as cur:
        cur.execute(f"SELECT status, output FROM {DB}.wf_step WHERE wf_id=%s AND step=%s", (wf_id, step))
        r = cur.fetchone()
    conn.commit()
    if r and r[0] == "COMPLETED":
        return True, (json.loads(r[1]) if r[1] else None)
    return False, None


def transaction(step):
    """Step whose body writes + checkpoint commit atomically (exactly-once)."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **k):
            conn, wf_id = _ctx.conn, _ctx.wf_id
            done, out = _step_done(conn, wf_id, step)
            if done:
                return out
            with conn.cursor() as cur:
                result = fn(cur, *a, **k)
                cur.execute(f"INSERT INTO {DB}.wf_step (wf_id,step,status,output) VALUES (%s,%s,'COMPLETED',%s)",
                            (wf_id, step, json.dumps(result)))
            conn.commit()
            return result
        return wrapper
    return deco


def step(step):
    """Non-transactional step (external side effect), checkpointed after success."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **k):
            conn, wf_id = _ctx.conn, _ctx.wf_id
            done, out = _step_done(conn, wf_id, step)
            if done:
                return out
            result = fn(*a, **k)
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO {DB}.wf_step (wf_id,step,status,output) VALUES (%s,%s,'COMPLETED',%s)",
                            (wf_id, step, json.dumps(result)))
            conn.commit()
            return result
        return wrapper
    return deco


def sleep(name, seconds, poll_timeout=25):
    """Durable timer: register a timer, the in-DB CREATE TASK poller fires it,
    resume on fire. Re-entrant after a crash (timer persists)."""
    conn, wf_id = _ctx.conn, _ctx.wf_id
    done, _ = _step_done(conn, wf_id, name)
    if done:
        return
    with conn.cursor() as cur:  # register the timer once (idempotent)
        cur.execute(f"SELECT status FROM {DB}.timers WHERE wf_id=%s AND step=%s", (wf_id, name))
        if cur.fetchone() is None:
            cur.execute(f"INSERT INTO {DB}.timers (wf_id,step,due_at,status) "
                        f"VALUES (%s,%s, date_add(now(), interval %s second),'PENDING')", (wf_id, name, seconds))
    conn.commit()
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        with conn.cursor() as cur:
            cur.execute(f"SELECT status FROM {DB}.timers WHERE wf_id=%s AND step=%s", (wf_id, name))
            st = cur.fetchone()[0]
        conn.commit()
        if st == "FIRED":
            break
        time.sleep(0.5)
    with conn.cursor() as cur:   # checkpoint the sleep as a completed step
        cur.execute(f"INSERT INTO {DB}.wf_step (wf_id,step,status,output) VALUES (%s,%s,'COMPLETED','null')",
                    (wf_id, name))
    conn.commit()


def current_attempt():
    return _ctx.attempt


def current_wf_id():
    return _ctx.wf_id


# ---------------- durable queue + workers ----------------
def enqueue(wf_name, wf_id, inp):
    c = _conn(autocommit=True)
    c.cursor().execute(f"INSERT INTO {DB}.dq (wf_name, wf_id, input, status) VALUES (%s,%s,%s,'READY')",
                       (wf_name, wf_id, json.dumps(inp)))
    c.cursor().execute(f"INSERT INTO {DB}.wf (wf_id,name,status,input) VALUES (%s,%s,'ENQUEUED',%s)",
                       (wf_id, wf_name, json.dumps(inp)))
    c.close()


def _claim(conn, worker):
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, wf_name, wf_id, input FROM {DB}.dq WHERE status='READY' "
                    f"ORDER BY id LIMIT 1 FOR UPDATE")
        row = cur.fetchone()
        if not row:
            conn.commit()
            return None
        did, wf_name, wf_id, inp = row
        cur.execute(f"UPDATE {DB}.dq SET status='CLAIMED', claimed_by=%s, attempts=attempts+1 WHERE id=%s",
                    (worker, did))
        cur.execute(f"SELECT attempts FROM {DB}.dq WHERE id=%s", (did,))
        attempt = cur.fetchone()[0]
    conn.commit()
    return did, wf_name, wf_id, json.loads(inp), attempt


def worker(worker_name, total, log):
    conn = _conn(autocommit=False)
    _ctx.conn = conn
    idle = 0
    while True:
        claimed = _claim(conn, worker_name)
        if claimed is None:
            done = _count(conn, "status='DONE'")
            if done >= total:
                break
            idle += 1
            if idle > 40:
                break
            time.sleep(0.3)
            continue
        idle = 0
        did, wf_name, wf_id, inp, attempt = claimed
        _ctx.wf_id, _ctx.attempt = wf_id, attempt
        try:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE {DB}.wf SET status='RUNNING' WHERE wf_id=%s", (wf_id,))
            conn.commit()
            out = _WORKFLOWS[wf_name](inp)
            with conn.cursor() as cur:
                cur.execute(f"UPDATE {DB}.wf SET status='COMPLETED', output=%s WHERE wf_id=%s",
                            (json.dumps(out), wf_id))
                cur.execute(f"UPDATE {DB}.dq SET status='DONE' WHERE id=%s", (did,))
            conn.commit()
            log.append(f"{worker_name} completed {wf_id} (attempt {attempt})")
        except Exception as e:
            conn.rollback()
            with conn.cursor() as cur:   # crash -> requeue for another attempt
                cur.execute(f"UPDATE {DB}.dq SET status='READY' WHERE id=%s", (did,))
            conn.commit()
            log.append(f"{worker_name} CRASH on {wf_id} (attempt {attempt}): {e} -> requeued")
    conn.close()


def run_workers(names, total):
    log = []
    threads = [threading.Thread(target=worker, args=(n, total, log)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return log


def _count(conn, where):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {DB}.dq WHERE {where}")
        n = cur.fetchone()[0]
    conn.commit()
    return n


def query(sql):
    c = _conn(autocommit=True)
    with c.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    c.close()
    return rows


def scalar(sql):
    r = query(sql)
    return r[0][0] if r else None
