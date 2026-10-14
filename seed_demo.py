"""Fill the database with ~350 demo complaints spread over the last 14 days (run once before the dashboard).
    python seed_demo.py
Complaints are pushed through the *real* pipeline (classification, urgency, duplicate linking, responses)."""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import generate_dataset as gd
from civicx import db
from civicx.config import SLA_HOURS
from civicx.pipeline import CivicX

random.seed(7); np.random.seed(7)
gd.random.seed(1234)                                   # different sample than the training corpus
df = gd.build(n_per_variant=6).sample(frac=1, random_state=3).reset_index(drop=True)

# "hotspots": a few issues reported by many citizens in EN + MR (-> duplicates / clusters / escalation)
HOT = [("Water Supply", 0, "Andheri"), ("Street Lighting", 0, "Dadar"), ("Drainage & Flooding", 2, "Kurla"),
       ("Sanitation", 1, "Govandi"), ("Roads", 0, "Malad"), ("Electricity", 3, "Bhandup")]
hot_rows = []
for cat, ti, loc in HOT:
    base, en_v, mr_v = gd.T[cat][ti]
    for k in range(16):
        if k % 2 == 0:
            s_ = random.choice(en_v).format(loc=loc); s_ = random.choice(gd.OPEN_EN) + s_
            s_ += random.choice([" It has been like this for %d days." % random.choice([2, 3, 4, 5]), "", " Please act urgently!"])
        else:
            s_ = random.choice(mr_v).format(loc=gd.LOC_MR_MAIN[loc]); s_ = random.choice(gd.OPEN_MR) + s_
            s_ += random.choice([" ही समस्या गेल्या %d दिवसांपासून सुरू आहे." % random.choice([2, 3, 4, 5]), "", " कृपया तातडीने कार्यवाही करा!"])
        hot_rows.append(dict(text=s_, category=cat))
pool = pd.concat([df.head(260)[["text", "category"]], pd.DataFrame(hot_rows)]).sample(frac=1, random_state=5).reset_index(drop=True)

cx = CivicX()
db.reset(cx.con)
now = datetime.now()
for _, r in pool.iterrows():
    # monsoon-style trend: water/drainage complaints are more recent
    recent_bias = r.category in ("Water Supply", "Drainage & Flooding")
    days_ago = np.random.beta(1.2, 3.5 if recent_bias else 1.6) * 14
    t = now - timedelta(days=float(days_ago), minutes=random.randint(0, 600))
    res = cx.submit(r.text, created_at=t)
    # resolve most tickets that are old enough
    age_h = (now - t).total_seconds() / 3600
    if not res["is_duplicate"] and age_h > 24 and random.random() < 0.6:
        took = SLA_HOURS[res["priority"]] * random.uniform(0.3, 1.3)
        if took < age_h:
            db.update(cx.con, res["id"], status="Resolved", resolved_at=(t + timedelta(hours=took)).isoformat(timespec="seconds"))
d = db.tickets_df(cx.con)
print(f"seeded {len(d)} complaints -> {(d.is_duplicate == 0).sum()} unique issues, "
      f"{(d.status == 'Resolved').sum()} resolved, {d.is_duplicate.sum()} linked duplicates")
