"""End-to-end CivicX pipeline:  complaint -> preprocess -> category -> urgency -> duplicate -> response -> DB"""
from datetime import datetime

import numpy as np

from . import db
from .config import CATEGORIES, DEPARTMENTS, DEPARTMENTS_MR
from .models import find_duplicate, load_artifacts
from .preprocess import preprocess
from .responder import generate_response
from .signals import explain


class CivicX:
    def __init__(self, db_path=None):
        art = load_artifacts()
        self.cat, self.urg, self.emb, self.thr = art["category"], art["urgency"], art["embedder"], art["dup_threshold"]
        self.con = db.connect(db_path)

    # -- analysis only (no DB write) -------------------------------------------------
    def analyze(self, text: str, lang: str = None) -> dict:
        p = preprocess(text, lang)
        top = self.cat.top_k(p, 3)
        priority, score, probs, sig = self.urg.predict([p])[0]
        return dict(processed=p, lang=p.lang, tokens=p.tokens, concepts=p.concepts, location=p.location,
                    category=top[0][0], cat_conf=top[0][1], top3=top, priority=priority, urgency_score=score,
                    priority_probs=probs, signals=sig, reasons=explain(sig, p.lang), embedding=self.emb.encode([p])[0])

    # -- full submission -----------------------------------------------------------------
    def submit(self, text: str, lang: str = None, created_at: datetime = None) -> dict:
        a = self.analyze(text, lang)
        created_at = created_at or datetime.now()
        match, sim = find_duplicate(a["embedding"], a["category"], a["location"], db.open_pool(self.con), self.thr)
        dept = (DEPARTMENTS_MR if a["lang"] == "mr" else DEPARTMENTS)[a["category"]]
        tid = db.insert_ticket(
            self.con, ticket_no="pending", created_at=created_at.isoformat(timespec="seconds"), text=text,
            lang=a["lang"], category=a["category"], cat_conf=a["cat_conf"], priority=a["priority"],
            urgency_score=a["urgency_score"], location=a["location"], department=dept, reasons=" | ".join(a["reasons"]),
            is_duplicate=int(match is not None), parent_id=match["id"] if match else None,
            dup_sim=sim if match else None, embedding=a["embedding"])
        ticket_no = f"CX-{created_at.year}-{tid:05d}"
        db.update(self.con, tid, ticket_no=ticket_no, cluster_id=match["id"] if match else tid)
        escalated = None
        n_dup = 0
        if match:
            n_dup = db.dup_count(self.con, match["id"]) + 1          # +1: the original ticket
            escalated = db.escalate_parent(self.con, match["id"])
        resp = generate_response(a["lang"], ticket_no, a["category"], a["priority"], a["location"],
                                 duplicate_of=match["ticket_no"] if match else None, dup_count=n_dup, created=created_at)
        db.update(self.con, tid, response=resp)
        a.update(id=tid, ticket_no=ticket_no, is_duplicate=match is not None, duplicate_of=match["ticket_no"] if match else None,
                 dup_sim=sim if match else None, dup_count=n_dup, response=resp, parent_escalated_to=escalated)
        return a
