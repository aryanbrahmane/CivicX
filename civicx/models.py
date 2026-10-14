"""Model layer: category classifier, urgency detector, sentence embedder, duplicate detection."""
import os
from typing import List

import joblib
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler

from .config import MODEL_DIR, PRIORITIES
from .preprocess import Processed
from .signals import extract_signals, signal_vector

PRIORITY_WEIGHT = np.array([0.10, 0.40, 0.75, 1.00])      # expected-severity weights -> urgency score


def feat_text(p: Processed) -> str:
    return p.text + " " + " ".join(p.concepts)


class _TextFeatures:
    """word 1-2gram TF-IDF on stemmed tokens (+concepts)  ||  char 2-4gram TF-IDF on the normalised text.
    The char n-grams make the model robust to Marathi morphology, typos and code-mixing."""

    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, token_pattern=r"(?u)\S+")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)

    def fit_transform(self, ps):
        return sparse.hstack([self.word.fit_transform([feat_text(p) for p in ps]),
                              self.char.fit_transform([p.normalized for p in ps])]).tocsr()

    def transform(self, ps):
        return sparse.hstack([self.word.transform([feat_text(p) for p in ps]),
                              self.char.transform([p.normalized for p in ps])]).tocsr()


class CategoryClassifier:
    """Module 1 - text classification (TF-IDF + Logistic Regression baseline)."""

    def __init__(self, C=10.0):
        self.feats = _TextFeatures()
        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, ps: List[Processed], y):
        self.clf.fit(self.feats.fit_transform(ps), y)
        return self

    def predict_proba(self, ps):
        return self.clf.predict_proba(self.feats.transform(ps))

    def predict(self, ps):
        return list(self.clf.classes_[self.predict_proba(ps).argmax(1)])

    def top_k(self, p, k=3):
        pr = self.predict_proba([p])[0]
        idx = pr.argsort()[::-1][:k]
        return [(self.clf.classes_[i], float(pr[i])) for i in idx]


class UrgencyDetector:
    """Module 2 - urgency detection. Text features + extracted signals (duration, impact, urgency words,
    severity lexicon, negation) -> 4-class Logistic Regression. Optional rule-based safety floor."""

    def __init__(self, C=5.0, hybrid=True):
        self.feats = _TextFeatures()
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=3000)
        self.hybrid = hybrid

    def _sig(self, ps):
        return [extract_signals(p.normalized, p.lang) for p in ps]

    def _matrix(self, ps, fit=False):
        sigs = self._sig(ps)
        num = np.array([signal_vector(s) for s in sigs])
        if fit:
            X_text = self.feats.fit_transform(ps)
            num = self.scaler.fit_transform(num)
        else:
            X_text = self.feats.transform(ps)
            num = self.scaler.transform(num)
        return sparse.hstack([X_text, sparse.csr_matrix(num * 2.0)]).tocsr(), sigs

    def fit(self, ps, y):
        X, _ = self._matrix(ps, fit=True)
        self.clf.fit(X, [PRIORITIES.index(v) for v in y])
        return self

    def predict_proba(self, ps):
        X, sigs = self._matrix(ps)
        pr = np.zeros((len(ps), 4))
        pr[:, self.clf.classes_] = self.clf.predict_proba(X)
        return pr, sigs

    @staticmethod
    def rule_floor(sig) -> int:
        """Safety net: never let clearly dangerous / hospital-related complaints fall below High."""
        if sig["hospital"] and (sig["duration_days"] >= 0.5 or sig["urgent_kw"]):
            return 2
        if sig["public_safety"] and sig["urgent_kw"]:
            return 2
        return 0

    def predict(self, ps):
        """-> list of (priority, score 0-1, probs, signals)"""
        pr, sigs = self.predict_proba(ps)
        out = []
        for row, sig in zip(pr, sigs):
            k = int(row.argmax())
            if self.hybrid:
                k = max(k, self.rule_floor(sig))
            score = float((row * PRIORITY_WEIGHT).sum())
            if self.hybrid and k > int(row.argmax()):
                score = max(score, PRIORITY_WEIGHT[k] * 0.9)
            out.append((PRIORITIES[k], score, row, sig))
        return out


class Embedder:
    """Sentence embeddings for duplicate detection.
      backend 'lsa'  : TF-IDF over stemmed tokens + cross-lingual concept/location tokens -> Truncated SVD (LSA)
                       (fully offline; English <-> Marathi duplicates are linked through the concept tokens)
      backend 'sbert': multilingual Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2) - set CIVICX_EMBEDDER=sbert
                       (needs `pip install sentence-transformers` and internet the first time)."""

    def __init__(self, backend=None, dim=96):
        self.backend = backend or os.environ.get("CIVICX_EMBEDDER", "lsa")
        self.dim = dim
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\S+")
        self.svd = None
        self.norm = Normalizer()
        self._sbert = None

    def _load_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self._sbert

    def fit(self, ps):
        if self.backend == "lsa":
            X = self.tfidf.fit_transform([p.semantic_text for p in ps])
            self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=0).fit(X)
        return self

    def encode(self, ps) -> np.ndarray:
        if self.backend == "sbert":
            return np.asarray(self._load_sbert().encode([p.raw for p in ps], normalize_embeddings=True), dtype="float32")
        Z = self.svd.transform(self.tfidf.transform([p.semantic_text for p in ps]))
        return self.norm.transform(Z).astype("float32")

    def __getstate__(self):
        d = self.__dict__.copy(); d["_sbert"] = None
        return d


def is_duplicate(sim: float, cat_a, cat_b, loc_a, loc_b, threshold: float) -> bool:
    """Semantic similarity + two sanity gates: same category, and not two *different* known localities."""
    if cat_a != cat_b:
        return False
    if loc_a and loc_b and loc_a != loc_b:
        return False
    return sim >= threshold


def find_duplicate(emb, category, location, pool, threshold):
    """pool: list of dicts {id, emb, category, location}. Returns (best_match_dict | None, best_sim)."""
    best, best_sim = None, 0.0
    for t in pool:
        s = float(np.dot(emb, t["emb"]))
        if is_duplicate(s, category, t["category"], location, t["location"], threshold) and s > best_sim:
            best, best_sim = t, s
    return best, best_sim


def cluster_complaints(embs: np.ndarray, categories, locations, threshold: float):
    """Offline clustering of complaints into 'issues': agglomerative (average-linkage, cosine) inside each
    (category, location) bucket - complaints in different buckets can never merge."""
    labels = -np.ones(len(embs), dtype=int)
    nxt = 0
    buckets = {}
    for i, (c, l) in enumerate(zip(categories, locations)):
        buckets.setdefault((c, l or "?"), []).append(i)
    for idx in buckets.values():
        if len(idx) == 1:
            labels[idx[0]] = nxt; nxt += 1; continue
        ag = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - threshold).fit(embs[idx])
        for j, lab in zip(idx, ag.labels_):
            labels[j] = nxt + lab
        nxt += ag.labels_.max() + 1
    return labels


def save_artifacts(cat, urg, emb, threshold, meta=None):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"category": cat, "urgency": urg, "embedder": emb, "dup_threshold": threshold, "meta": meta or {}},
                MODEL_DIR / "civicx_models.joblib")


def load_artifacts():
    return joblib.load(MODEL_DIR / "civicx_models.joblib")
