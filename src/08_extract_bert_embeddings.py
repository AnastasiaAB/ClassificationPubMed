"""
Step 8: Extract frozen BiomedBERT embeddings for every abstract (PyTorch + transformers).

"Frozen" means the pretrained model is used as a fixed feature extractor: no training,
no weight updates. Each abstract (title + abstract, same text as the other models) is
turned into one 768-dimensional vector, which 09_train_bert_head.py classifies.

Design choices (fixed in advance, not tuned against the results):
- Model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
  (formerly PubMedBERT; pretrained from scratch on PubMed abstracts + PMC full text)
- Max length 512 tokens (the model's limit); longer texts are truncated
- Pooling: mean of the final-layer token vectors, ignoring padding

Why this is safe to run on ALL abstracts at once: the encoder never sees labels, so
computing embeddings for the test set cannot leak label information.

CPU-friendly: texts are processed shortest-first so batches carry little padding, and
progress is checkpointed so an interrupted run resumes where it stopped.
Expect roughly tens of minutes on a laptop CPU (not measured by me).

Output: data/processed/bert_embeddings.npz (embeddings, pmids, model name, settings)
"""

import sqlite3
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from config import DB_PATH, EMBEDDINGS_PATH

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512
BATCH_SIZE = 16
CHECKPOINT_EVERY = 10  # batches
PARTIAL_PATH = EMBEDDINGS_PATH.with_name("bert_embeddings_partial.npz")


def load_texts():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT pmid, title, abstract FROM abstracts", conn)
    conn.close()
    texts = (df["title"].fillna("") + " " + df["abstract"]).tolist()
    return df["pmid"].to_numpy(), texts


def save(path, embeddings, done, pmids):
    np.savez(
        path,
        embeddings=embeddings, done=done, pmids=pmids,
        model_name=np.array(MODEL_NAME), max_length=np.array(MAX_LENGTH),
        pooling=np.array("mean"),
    )


def main():
    pmids, texts = load_texts()
    n = len(texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{n} abstracts | model: {MODEL_NAME} | device: {device}")

    # First run downloads the model weights (a few hundred MB) into the Hugging Face cache
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
    lengths = np.array([len(ids) for ids in enc["input_ids"]])
    print(f"token length: median={int(np.median(lengths))}, max={lengths.max()}, "
          f"truncated at {MAX_LENGTH}: {(lengths >= MAX_LENGTH).sum()} abstracts")
    order = np.argsort(lengths)  # shortest first -> minimal padding per batch

    embeddings = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    done = np.zeros(n, dtype=bool)

    # Resume from an interrupted run if the checkpoint matches this exact setup
    if PARTIAL_PATH.exists():
        part = np.load(PARTIAL_PATH)
        same = (np.array_equal(part["pmids"], pmids)
                and str(part["model_name"]) == MODEL_NAME
                and int(part["max_length"]) == MAX_LENGTH)
        if same:
            embeddings, done = part["embeddings"], part["done"]
            print(f"Resuming from checkpoint: {done.sum()}/{n} already done")
        else:
            print("Found a checkpoint from a different setup; ignoring it")

    batches = [order[i:i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    start = time.time()
    for b, idx in enumerate(batches, start=1):
        if done[idx].all():
            continue
        features = [{k: enc[k][i] for k in enc.keys()} for i in idx]
        batch = tokenizer.pad(features, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            last_hidden = model(**batch).last_hidden_state          # (batch, tokens, 768)
        mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings[idx] = pooled.cpu().numpy()
        done[idx] = True

        if b % CHECKPOINT_EVERY == 0:
            save(PARTIAL_PATH, embeddings, done, pmids)
            print(f"  {done.sum()}/{n} done ({done.mean():.0%}), {time.time() - start:.0f}s elapsed "
                  f"(later batches are longer, so remaining time is more than a linear estimate)")

    assert done.all(), "some abstracts were not embedded"
    assert np.isfinite(embeddings).all(), "found NaN/inf in embeddings"
    save(EMBEDDINGS_PATH, embeddings, done, pmids)
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
    print(f"Saved {embeddings.shape} embeddings -> {EMBEDDINGS_PATH} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
