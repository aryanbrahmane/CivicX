"""Train + evaluate every CivicX model.   python train.py
Outputs: models/civicx_models.joblib, models/metrics.json, models/*.png"""
import json
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from civicx.config import CATEGORIES, DATASET_PATH, DATA_DIR, MODEL_DIR, PRIORITIES
from civicx.models import (CategoryClassifier, Embedder, UrgencyDetector, is_duplicate, save_artifacts, cluster_complaints)
from civicx.preprocess import preprocess

warnings.filterwarnings("ignore")
random.seed(0); np.random.seed(0)
MODEL_DIR.mkdir(exist_ok=True)
metrics = {}

df = pd.read_csv(DATASET_PATH)
print(f"Dataset: {len(df)} complaints | languages: {df.lang.value_counts().to_dict()}")
P = [preprocess(t, l) for t, l in zip(df.text, df.lang)]        # language-specific preprocessing
df["_p"] = P
hw = pd.read_csv(DATA_DIR / "handwritten_test.csv")
HP = [preprocess(t) for t in hw.text]                            # language auto-detected here
print("language detection accuracy on hand-written set:", accuracy_score(hw.lang, [p.lang for p in HP]))

tr, te = train_test_split(df, test_size=0.2, stratify=df.category + df.lang, random_state=0)

# ----------------------------------------------------------------- Module 1: category
print("\n=== Module 1: category classification ===")
def feats_for(tr_ps, te_ps, kind):
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\S+") if kind == "word" else \
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    f = (lambda ps: [p.text + " " + " ".join(p.concepts) for p in ps]) if kind == "word" else (lambda ps: [p.normalized for p in ps])
    return v.fit_transform(f(tr_ps)), v.transform(f(te_ps))

comp = {}
Xw_tr, Xw_te = feats_for(tr._p, te._p, "word")
for name, clf in [("TF-IDF(word) + MultinomialNB", MultinomialNB(alpha=0.1)),
                  ("TF-IDF(word) + LinearSVC", LinearSVC(C=1.0)),
                  ("TF-IDF(word) + LogReg", LogisticRegression(C=10, max_iter=2000))]:
    clf.fit(Xw_tr, tr.category); comp[name] = f1_score(te.category, clf.predict(Xw_te), average="macro")
cat = CategoryClassifier().fit(list(tr._p), tr.category)
comp["TF-IDF(word+char) + LogReg  [CivicX]"] = f1_score(te.category, cat.predict(list(te._p)), average="macro")
for k, v in comp.items(): print(f"  {k:42s} macro-F1 = {v:.3f}")
metrics["category_model_comparison_macroF1"] = comp

pred = cat.predict(list(te._p))
print(classification_report(te.category, pred, digits=3))
metrics["category_random_split_acc"] = accuracy_score(te.category, pred)
for lg in ["en", "mr"]:
    m = te.lang == lg
    metrics[f"category_random_split_acc_{lg}"] = accuracy_score(te.category[m], np.array(pred)[m.values])
    print(f"  accuracy [{lg}] = {metrics[f'category_random_split_acc_{lg}']:.3f}")

# stricter test 1: unseen paraphrase (train on paraphrase 0, test on paraphrase 1)
a, b = df[df.variant == 0], df[df.variant == 1]
c2 = CategoryClassifier().fit(list(a._p), a.category)
metrics["category_unseen_paraphrase_acc"] = accuracy_score(b.category, c2.predict(list(b._p)))
print(f"  UNSEEN-PARAPHRASE accuracy = {metrics['category_unseen_paraphrase_acc']:.3f}")
# ablation: cross-lingual transfer (train on English only, test on Marathi) - shows the value of concept tokens
en, mr_ = df[df.lang == "en"], df[df.lang == "mr"]
c3 = CategoryClassifier().fit(list(en._p), en.category)
metrics["category_en_to_mr_transfer_acc"] = accuracy_score(mr_.category, c3.predict(list(mr_._p)))
print(f"  ZERO-SHOT English->Marathi transfer accuracy = {metrics['category_en_to_mr_transfer_acc']:.3f}")
# stricter test 2: independent hand-written complaints
hp = cat_all = CategoryClassifier().fit(list(df._p), df.category)      # final model (all data)
hpred = cat_all.predict(HP)
metrics["category_handwritten_acc"] = accuracy_score(hw.category, hpred)
for lg in ["en", "mr"]:
    m = (hw.lang == lg).values
    metrics[f"category_handwritten_acc_{lg}"] = accuracy_score(hw.category[m], np.array(hpred)[m])
print(f"  HAND-WRITTEN set accuracy = {metrics['category_handwritten_acc']:.3f} "
      f"(en {metrics['category_handwritten_acc_en']:.3f}, mr {metrics['category_handwritten_acc_mr']:.3f})")
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(te.category, pred, xticks_rotation=45, ax=ax, colorbar=False)
plt.tight_layout(); plt.savefig(MODEL_DIR / "category_confusion.png", dpi=130); plt.close()

# ----------------------------------------------------------------- Module 2: urgency
print("\n=== Module 2: urgency detection ===")
urg = UrgencyDetector(hybrid=False).fit(list(tr._p), tr.priority)
out = urg.predict(list(te._p)); up = [o[0] for o in out]
metrics["urgency_random_split_acc"] = accuracy_score(te.priority, up)
metrics["urgency_random_split_macroF1"] = f1_score(te.priority, up, average="macro")
print(classification_report(te.priority, up, labels=PRIORITIES, digits=3))
within1 = np.mean([abs(PRIORITIES.index(a) - PRIORITIES.index(b)) <= 1 for a, b in zip(te.priority, up)])
metrics["urgency_within_one_level"] = float(within1)
print(f"  within-one-level accuracy = {within1:.3f}")
# baseline: keyword-only rule (what many orgs use today)
def kw_rule(p):
    s = p.normalized
    return "Critical" if any(k in s for k in ["urgent", "emergency", "तातडी", "आणीबाणी"]) else "Medium"
metrics["urgency_keyword_baseline_acc"] = accuracy_score(te.priority, [kw_rule(p) for p in te._p])
print(f"  keyword-rule baseline accuracy = {metrics['urgency_keyword_baseline_acc']:.3f}")
ub, _ = df[df.variant == 0], df[df.variant == 1]
u2 = UrgencyDetector(hybrid=False).fit(list(ub._p), ub.priority)
bb = df[df.variant == 1]
metrics["urgency_unseen_paraphrase_acc"] = accuracy_score(bb.priority, [o[0] for o in u2.predict(list(bb._p))])
print(f"  UNSEEN-PARAPHRASE accuracy = {metrics['urgency_unseen_paraphrase_acc']:.3f}")
urg_all_plain = UrgencyDetector(hybrid=False).fit(list(df._p), df.priority)
urg_all = UrgencyDetector(hybrid=True).fit(list(df._p), df.priority)
for name, m in [("model only", urg_all_plain), ("model + rule floor", urg_all)]:
    o = [x[0] for x in m.predict(HP)]
    acc = accuracy_score(hw.priority, o); w1 = np.mean([abs(PRIORITIES.index(a) - PRIORITIES.index(b)) <= 1 for a, b in zip(hw.priority, o)])
    metrics[f"urgency_handwritten_{name.replace(' ', '_')}"] = {"acc": acc, "within_one": float(w1)}
    print(f"  HAND-WRITTEN [{name}] accuracy = {acc:.3f} | within-one-level = {w1:.3f}")

# ----------------------------------------------------------------- Module 3: duplicates
print("\n=== Module 3: duplicate / cluster detection ===")
idx = np.arange(len(df)); np.random.shuffle(idx)
n = len(idx); fit_i, val_i, test_i = idx[: int(.6 * n)], idx[int(.6 * n): int(.8 * n)], idx[int(.8 * n):]
emb = Embedder().fit([P[i] for i in fit_i])

def make_pairs(ids, k=4000):
    ids = list(ids); by_issue = {}
    for i in ids: by_issue.setdefault(df.issue_id[i], []).append(i)
    by_cat = {}
    for i in ids: by_cat.setdefault(df.category[i], []).append(i)
    pairs = []
    for _ in range(k // 2):                       # positives: same issue (same problem, same place) incl. cross-lingual
        g = random.choice([v for v in by_issue.values() if len(v) > 1]); a, b = random.sample(g, 2); pairs.append((a, b, 1))
    by_cl = {}
    for i in ids: by_cl.setdefault((df.category[i], df.location[i]), []).append(i)
    by_cl = [v for v in by_cl.values() if len(v) > 1]
    while len(pairs) < k:                         # hard negatives: same category AND same locality, different problem
        if random.random() < 0.8:
            a, b = random.sample(random.choice(by_cl), 2)
        else:                                     # easy negatives: same category anywhere
            a, b = random.sample(by_cat[random.choice(list(by_cat))], 2)
        if df.issue_id[a] != df.issue_id[b]: pairs.append((a, b, 0))
    return pairs

def sims(pairs):
    E = emb.encode([P[i] for p in pairs for i in p[:2]]).reshape(len(pairs), 2, -1)
    return (E[:, 0] * E[:, 1]).sum(1)

vp, tp = make_pairs(val_i), make_pairs(test_i)
vs, ts = sims(vp), sims(tp)
vy, ty = np.array([p[2] for p in vp]), np.array([p[2] for p in tp])
gate = lambda pairs, s, thr: np.array([is_duplicate(x, df.category[a], df.category[b], df.location[a], df.location[b], thr)
                                       for (a, b, _), x in zip(pairs, s)]).astype(int)
best_thr, best_f1 = 0.7, -1
for thr in np.arange(0.2, 0.98, 0.01):
    f = f1_score(vy, gate(vp, vs, thr))
    if f > best_f1: best_thr, best_f1 = float(thr), f
print(f"  threshold tuned on validation pairs = {best_thr:.2f} (F1={best_f1:.3f})")
pg = gate(tp, ts, best_thr); pc = (ts >= best_thr).astype(int)
for name, pr in [("cosine only", pc), ("cosine + category/location gate", pg)]:
    p_, r_, f_, _ = precision_recall_fscore_support(ty, pr, average="binary")
    metrics[f"dup_{name.replace(' ', '_').replace('/', '_')}"] = dict(precision=p_, recall=r_, f1=f_)
    print(f"  [{name:32s}] precision={p_:.3f} recall={r_:.3f} F1={f_:.3f}")
cross = np.array([df.lang[a] != df.lang[b] for a, b, _ in tp]) & (ty == 1)
metrics["dup_cross_lingual_recall"] = float(pg[cross].mean()); metrics["dup_same_language_recall"] = float(pg[(ty == 1) & ~cross].mean())
print(f"  recall same-language pairs = {metrics['dup_same_language_recall']:.3f} | cross-lingual (EN<->MR) = {metrics['dup_cross_lingual_recall']:.3f}")
# clustering quality on a 600-complaint sample
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
samp = test_i[:600]
lab = cluster_complaints(emb.encode([P[i] for i in samp]), [df.category[i] for i in samp], [df.location[i] for i in samp], best_thr)
metrics["cluster_ARI"] = adjusted_rand_score([df.issue_id[i] for i in samp], lab)
print(f"  clustering of {len(samp)} complaints -> {len(set(lab))} clusters (true issues: {df.issue_id[samp].nunique()}), "
      f"ARI={metrics['cluster_ARI']:.3f}, homogeneity={homogeneity_score(df.issue_id[samp], lab):.3f}")

# ----------------------------------------------------------------- save final artifacts (fit on all data)
emb_all = Embedder().fit(P)
save_artifacts(cat_all, urg_all, emb_all, best_thr, meta={"n_train": len(df), "categories": CATEGORIES})
json.dump(metrics, open(MODEL_DIR / "metrics.json", "w"), indent=2, default=float)
print("\nSaved models/civicx_models.joblib and models/metrics.json")
