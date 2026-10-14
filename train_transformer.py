"""OPTIONAL: fine-tune a multilingual transformer for category classification (needs GPU/internet; NOT run in the
sandbox where CivicX was built - the TF-IDF model in train.py is the tested default).
    pip install transformers torch datasets accelerate
    python train_transformer.py            # MuRIL (Google) understands Marathi + English; alternative: xlm-roberta-base
"""
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)

from civicx.config import CATEGORIES, DATASET_PATH, MODEL_DIR

MODEL = "google/muril-base-cased"
df = pd.read_csv(DATASET_PATH)
df["label"] = df.category.map({c: i for i, c in enumerate(CATEGORIES)})
tr, te = train_test_split(df, test_size=0.2, stratify=df.category, random_state=0)
tok = AutoTokenizer.from_pretrained(MODEL)
enc = lambda b: tok(b["text"], truncation=True, max_length=128, padding="max_length")
ds_tr = Dataset.from_pandas(tr[["text", "label"]]).map(enc, batched=True)
ds_te = Dataset.from_pandas(te[["text", "label"]]).map(enc, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(CATEGORIES))
args = TrainingArguments(output_dir=str(MODEL_DIR / "transformer"), num_train_epochs=3, per_device_train_batch_size=16,
                         learning_rate=3e-5, eval_strategy="epoch", save_strategy="no", report_to=[])
compute = lambda p: {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}
Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te, compute_metrics=compute).train()
model.save_pretrained(MODEL_DIR / "transformer"); tok.save_pretrained(MODEL_DIR / "transformer")
