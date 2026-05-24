"""Thin pymysql wrapper for MatrixOne.

MatrixOne speaks the MySQL wire protocol; the only quirk is the username,
which is `account:user:role`. pymysql passes it through verbatim.
"""
import pymysql

import config


class MO:
    def __init__(self, database=None):
        p = config.mo_conn_params()
        self._params = dict(
            host=p["host"],
            port=p["port"],
            user=p["user"],
            password=p["password"],
            autocommit=True,
            charset="utf8mb4",
            local_infile=False,
        )
        if database:
            self._params["database"] = database
        self.conn = pymysql.connect(**self._params)

    def query(self, sql, args=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def query_one(self, sql, args=None):
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def scalar(self, sql, args=None):
        row = self.query_one(sql, args)
        return row[0] if row else None

    def execute(self, sql, args=None):
        with self.conn.cursor() as cur:
            return cur.execute(sql, args)

    def executemany(self, sql, seq):
        with self.conn.cursor() as cur:
            return cur.executemany(sql, seq)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
