"""Module 5 helpers - pure pandas analytics behind the dashboard (testable without Streamlit)."""
from datetime import datetime, timedelta

import pandas as pd

from .config import PRIORITIES


def kpis(df: pd.DataFrame) -> dict:
    parents = df[df.is_duplicate == 0]
    resolved = parents[parents.status == "Resolved"]
    hrs = ((resolved.resolved_at - resolved.created_at).dt.total_seconds() / 3600).mean() if len(resolved) else float("nan")
    open_p = parents[parents.status != "Resolved"]
    return dict(total=len(df), unique_issues=len(parents), duplicates_merged=int(df.is_duplicate.sum()),
                open_issues=len(open_p), critical_open=int((open_p.priority == "Critical").sum()),
                avg_response_hours=hrs,
                priority_counts={p: int((open_p.priority == p).sum()) for p in PRIORITIES[::-1]})


def trending(df: pd.DataFrame, now=None, days=7) -> pd.DataFrame:
    """Category volume in the last `days` vs the `days` before -> % change (trend arrows)."""
    now = now or df.created_at.max()
    cur = df[(df.created_at > now - timedelta(days=days))].category.value_counts()
    prev = df[(df.created_at <= now - timedelta(days=days)) & (df.created_at > now - timedelta(days=2 * days))].category.value_counts()
    t = pd.DataFrame({"last_period": cur, "previous_period": prev}).fillna(0).astype(int)
    t["change_pct"] = ((t.last_period - t.previous_period) / t.previous_period.replace(0, 1) * 100).round(0)
    t["trend"] = t.change_pct.apply(lambda x: "↑" if x > 10 else ("↓" if x < -10 else "→"))
    return t.sort_values("change_pct", ascending=False).reset_index().rename(columns={"index": "category"})


def cluster_table(df: pd.DataFrame, top=15) -> pd.DataFrame:
    """One row per issue (cluster) with number of citizens who reported it."""
    g = df.groupby("cluster_id")
    out = g.agg(reports=("id", "count"), category=("category", "first"), location=("location", "first"),
                first_reported=("created_at", "min"), last_reported=("created_at", "max"),
                languages=("lang", lambda s: "+".join(sorted(set(s)))))
    parent = df[df.is_duplicate == 0].set_index("cluster_id")
    out["priority"] = parent.priority
    out["ticket_no"] = parent.ticket_no
    out["sample_complaint"] = parent.text.str.slice(0, 110)
    return out.sort_values("reports", ascending=False).head(top).reset_index()


def hotspots(df: pd.DataFrame, top=10) -> pd.DataFrame:
    d = df.dropna(subset=["location"])
    return d.groupby("location").size().sort_values(ascending=False).head(top).rename("complaints").reset_index()
