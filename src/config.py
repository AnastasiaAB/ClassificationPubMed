"""Shared paths and constants so every script runs from the repo root regardless of machine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "pubmed_abstracts.csv"
DB_PATH = ROOT / "data" / "processed" / "pubmed.db"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

for _d in (DB_PATH.parent, RESULTS, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
EMBEDDINGS_PATH = DB_PATH.parent / "bert_embeddings.npz"
