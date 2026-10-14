"""SQLite persistence for tickets."""
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DB_PATH, PRIORITIES

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_no TEXT, created_at TEXT, text TEXT, lang TEXT,
  category TEXT, cat_conf REAL, priority TEXT, urgency_score REAL, location TEXT, department TEXT,
  reasons TEXT, cluster_id INTEGER, parent_id INTEGER, is_duplicate INTEGER DEFAULT 0, dup_sim REAL,
  response TEXT, status TEXT DEFAULT 'Open', resolved_at TEXT, embedding BLOB);
"""


def connect(path=None):
    con = sqlite3.connect(str(path or DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def insert_ticket(con, **f):
    f.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    emb = f.pop("embedding", None)
    cols = list(f) + ["embedding"]
    vals = list(f.values()) + [None if emb is None else np.asarray(emb, dtype="float32").tobytes()]
    cur = con.execute(f"INSERT INTO tickets({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)
    con.commit()
    return cur.lastrowid


def update(con, tid, **f):
    con.execute(f"UPDATE tickets SET {','.join(k + '=?' for k in f)} WHERE id=?", list(f.values()) + [tid])
    con.commit()


def open_pool(con):
    """Open *parent* tickets (one per issue) with embeddings -> candidates for duplicate matching."""
    rows = con.execute("SELECT id, ticket_no, category, location, embedding FROM tickets "
                       "WHERE is_duplicate=0 AND status!='Resolved' AND embedding IS NOT NULL").fetchall()
    return [dict(id=r["id"], ticket_no=r["ticket_no"], category=r["category"], location=r["location"],
                 emb=np.frombuffer(r["embedding"], dtype="float32")) for r in rows]


def dup_count(con, parent_id):
    return con.execute("SELECT COUNT(*) FROM tickets WHERE parent_id=?", (parent_id,)).fetchone()[0]


def escalate_parent(con, parent_id, min_reports=5):
    """Widespread problem (many citizens reporting) -> bump priority of the parent ticket by one level."""
    n = dup_count(con, parent_id)
    row = con.execute("SELECT priority FROM tickets WHERE id=?", (parent_id,)).fetchone()
    if n + 1 >= min_reports and row and row["priority"] != "Critical":
        k = PRIORITIES.index(row["priority"]) + 1
        # only escalate once per threshold crossing: at exactly min_reports and again at 4*min_reports
        if (n + 1) in (min_reports, 4 * min_reports):
            update(con, parent_id, priority=PRIORITIES[k])
            return PRIORITIES[k]
    return None


def tickets_df(con):
    df = pd.read_sql_query("SELECT id,ticket_no,created_at,text,lang,category,cat_conf,priority,urgency_score,location,"
                           "department,reasons,cluster_id,parent_id,is_duplicate,dup_sim,response,status,resolved_at "
                           "FROM tickets", con)
    for c in ("created_at", "resolved_at"):
        df[c] = pd.to_datetime(df[c])
    return df


def reset(con):
    con.execute("DELETE FROM tickets"); con.execute("DELETE FROM sqlite_sequence WHERE name='tickets'"); con.commit()
