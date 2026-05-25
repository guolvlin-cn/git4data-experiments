"""Demo app for the minimal DBOS-on-MatrixOne framework (durable_exec/framework.py).

An order workflow written with @workflow / @transaction / @step decorators + a
durable dbos.sleep(); orders are enqueued and drained by 2 concurrent workers.
One order is made to crash after charging on its first attempt — it is requeued
and resumed (completed steps skipped), so every side effect happens exactly once.

Run:  python3 -m durable_exec.app
"""
import durable_exec.framework as dbos


@dbos.transaction("reserve_inventory")
def reserve(cur, order):
    cur.execute(f"UPDATE {dbos.DB}.inventory SET qty = qty - %s WHERE sku = %s",
                (order["qty"], order["sku"]))
    return {"reserved": order["qty"]}


@dbos.transaction("charge_payment")
def charge(cur, order):
    cur.execute(f"INSERT INTO {dbos.DB}.payments (wf_id, amount) VALUES (%s, %s)",
                (dbos.current_wf_id(), order["amount"]))
    return {"charged": order["amount"]}


@dbos.transaction("ship_order")
def ship(cur, order):
    return {"shipped": order["sku"]}


@dbos.step("send_receipt")        # non-transactional external side effect
def notify(order):
    dbos.query(f"INSERT INTO {dbos.DB}.emails (wf_id, body) VALUES "
               f"('{dbos.current_wf_id()}', 'receipt for {order['sku']}')")
    return {"emailed": True}


@dbos.workflow("order")
def order_workflow(order):
    reserve(order)
    charge(order)
    if order.get("crash") and dbos.current_attempt() == 1:
        raise RuntimeError("injected crash after charge")
    dbos.sleep("settle_window", 2)     # durable timer (fired by the in-DB CREATE TASK)
    ship(order)
    notify(order)
    return {"status": "done", "sku": order["sku"]}


def hr(s):
    print("\n" + "=" * 72 + f"\n  {s}\n" + "=" * 72)


def main():
    dbos.setup()
    orders = [
        ("order-1", {"sku": "WIDGET", "qty": 1, "amount": 19.99}),
        ("order-2", {"sku": "WIDGET", "qty": 1, "amount": 29.99, "crash": True}),
        ("order-3", {"sku": "WIDGET", "qty": 2, "amount": 39.99}),
    ]
    hr("Enqueue 3 order workflows; 2 concurrent workers drain the durable queue")
    for wf_id, o in orders:
        dbos.enqueue("order", wf_id, o)
    print("  enqueued: " + ", ".join(wf for wf, _ in orders) + "  (order-2 crashes on attempt 1)")
    print("  durable sleep(2s) per order is fired by the in-DB CREATE TASK scheduler ...")

    log = dbos.run_workers(["worker-1", "worker-2"], total=len(orders))

    hr("Worker log (crash -> requeue -> resume)")
    for line in log:
        print("   ", line)

    hr("Verify exactly-once + durable completion")
    statuses = dict(dbos.query(f"SELECT wf_id, status FROM {dbos.DB}.wf ORDER BY wf_id"))
    payments = int(dbos.scalar(f"SELECT COUNT(*) FROM {dbos.DB}.payments"))
    inv = int(dbos.scalar(f"SELECT qty FROM {dbos.DB}.inventory WHERE sku='WIDGET'"))
    emails = int(dbos.scalar(f"SELECT COUNT(*) FROM {dbos.DB}.emails"))
    fired = int(dbos.scalar(f"SELECT COUNT(*) FROM {dbos.DB}.timers WHERE status='FIRED'"))
    print(f"  workflow statuses: {statuses}")
    print(f"  payments rows={payments} (expect 3, charge exactly-once even for the crashed one)")
    print(f"  inventory qty={inv} (expect 100-(1+1+2)=96, reserve exactly-once)")
    print(f"  receipt emails={emails} (expect 3); durable timers FIRED={fired} (expect 3)")
    ok = (all(s == "COMPLETED" for s in statuses.values()) and payments == 3 and inv == 96
          and emails == 3 and fired == 3)
    print(f"  ALL EXACTLY-ONCE & COMPLETE: {ok}")

    hr("Durable step log for the crashed workflow (order-2)")
    for stp, st in dbos.query(f"SELECT step, status FROM {dbos.DB}.wf_step WHERE wf_id='order-2' ORDER BY step"):
        print(f"    {stp:<18} {st}")
    print("    (reserve/charge were checkpointed on attempt 1 -> skipped on attempt 2; "
          "no double charge/decrement)")

    dbos.teardown()
    hr("Done — one DBOS-on-MatrixOne engine: @workflow/@transaction/@step + durable timer + durable queue")
    print("  Decorators give the programming model; @transaction commits side effect + checkpoint")
    print("  atomically (exactly-once); dbos.sleep is a durable timer fired by CREATE TASK; the")
    print("  queue + FOR UPDATE dispatch workflows to workers; a crash is requeued and resumed by")
    print("  skipping completed steps. All state + scheduling live inside MatrixOne.")


if __name__ == "__main__":
    main()
