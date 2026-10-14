# CivicX — End-to-End NLP Framework for Automated Complaint Triage and Clustering
Bilingual (**English + मराठी**) · Mumbai civic complaints [BMC] · category · urgency · duplicate clustering · auto-reply · dashboard

## Run it
```bash
pip install -r requirements.txt
python generate_dataset.py      # builds data/mumbai_complaints.csv (4,224 EN/MR complaints)
python train.py                 # trains + evaluates, saves models/civicx_models.joblib, metrics.json, confusion matrix
python seed_demo.py             # (optional) loads ~350 demo complaints into SQLite through the real pipeline
python -m tests.test_smoke      # sanity tests
streamlit run app.py            # dashboard
```

## Architecture
```
Complaint (EN / MR / code-mixed)
   │  detect_language  (Devanagari-ratio)
   ▼
Language-specific preprocessing            civicx/preprocess.py
   EN: normalise → tokenise → stop-words → lemma table + suffix stemmer
   MR: NFC/ZWJ/digit/anusvara normalisation → tokenise → stop-words → dictionary lemmatiser
       + longest-suffix stemmer (मध्ये, ात, ाला, ांच्या, पासून, ...) + oblique-form rule (कुर्ल्य→कुर्ल)
   + gazetteer NER for 25 Mumbai localities (EN + MR forms)
   + cross-lingual "concept tokens" (WATER, POTHOLE, SEWAGE …) from a bilingual lexicon
   ▼
Module 1 Category  ── TF-IDF(word 1-2g) ‖ TF-IDF(char 2-4g) → Logistic Regression   (8 classes)
Module 2 Urgency   ── same text features + extracted signals (duration, impact, urgency words,
                      severity lexicon, negation) → 4-class LR → urgency score 0-100 % + reasons
                      + rule-based safety floor (hospital / public-safety)
Module 3 Duplicates── sentence embeddings (LSA offline | multilingual SBERT optional) → cosine
                      + gates (same category, same locality) → online linking + agglomerative clustering
Module 4 Response  ── bilingual template reply (ticket ID, department, SLA); replies in the complaint's language
Module 5 Dashboard ── SQLite + pandas + Plotly + Streamlit (KPIs, trends, hotspots, clusters, tickets)
```

## NLP concepts
| Concept | Where |
|---|---|
| Unicode normalisation, script handling | `preprocess.normalize` |
| Tokenisation, stop-word removal (negation kept) | `preprocess.preprocess` |
| Stemming (rule-based, Marathi & English) + dictionary lemmatisation | `stem_mr`, `stem_en`, `_MR_LEMMA` |
| Language identification | `detect_language` |
| Named-entity recognition (gazetteer) | `extract_location` |
| Information extraction (regex): duration, impact, urgency | `signals.py` |
| TF-IDF, word + character n-grams | `models._TextFeatures` |
| Text classification (NB / SVM / LR compared) | `train.py` |
| Lexicon-based severity/sentiment cues, explainability | `signals.explain` |
| Latent Semantic Analysis / sentence embeddings, cosine similarity | `models.Embedder` |
| Cross-lingual matching via concept lexicon (and SBERT option) | `lexicon.CONCEPTS` |
| Clustering (agglomerative, cosine) | `models.cluster_complaints` |
| NLG (template-based, bilingual) | `responder.py` |

## Results
| Test | Result |
|---|---|
| Category, random 80/20 split | 100 % (**inflated** – corpus is template based) |
| Category, **unseen paraphrase** | 96.4 % |
| Category, zero-shot English-trained → Marathi test | 91.3 % |
| Category, **48 independent hand-written complaints** | 91.7 % (EN 87.5 %, MR 95.8 %) |
| Urgency, random split | 90.2 % (keyword-rule baseline: 32.8 %) |
| Urgency, unseen paraphrase | 66.8 % |
| Urgency, hand-written set | 52 % exact, 90 % within one level |
| Duplicate pairs (hard negatives) | P 0.83 / R 0.83 / F1 0.83 |
| Duplicate recall: same language / EN↔MR | 98 % / 70 % |
