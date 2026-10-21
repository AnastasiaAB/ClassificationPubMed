"""
Step 2: Clean PubMed abstract data and load into SQLite.

Input:  pubmed_abstracts.csv  (columns: pmid, title, abstract, journal, year, label)
Output: pubmed.db             (SQLite database, table `abstracts`)

Data quality issues found during EDA and how each is handled:

1. Duplicate PMIDs with CONFLICTING labels (58 articles)
   These papers were pulled under both MeSH category queries (e.g. a
   cardio-oncology paper indexed under both Neoplasms and Cardiovascular
   Diseases). Since ground truth is genuinely ambiguous for a binary
   classifier, these are DROPPED rather than arbitrarily assigned a label.
   This is worth a line in the README/write-up as a deliberate modeling
   decision, not an oversight.

2. Truncated/corrupted abstracts (abstract length < MIN_ABSTRACT_LEN)
   A handful of abstracts were cut short during the pull (e.g. one is just
   "The"), almost certainly from special characters (Greek letters, math
   notation, gene symbol superscripts) breaking the parser. These carry no
   usable signal and are DROPPED.

3. Missing titles / years
   Titles and years are not used as model features here, so nulls are kept
   as NULL in SQLite rather than dropping rows or imputing.

4. Whitespace / encoding cleanup
   Abstracts and titles are stripped of leading/trailing whitespace and
   collapsed internal whitespace (some records had irregular spacing from
   the pull).
"""

import sqlite3
import pandas as pd
from config import RAW_CSV, DB_PATH

INPUT_CSV = RAW_CSV
OUTPUT_DB = DB_PATH
MIN_ABSTRACT_LEN = 100  # chars; below this, abstract is treated as corrupted

def clean_whitespace(series: pd.Series) -> pd.Series:
    return series.str.strip().str.replace(r"\s+", " ", regex=True)

def main():
    df = pd.read_csv(INPUT_CSV)
    n_start = len(df)

    # --- 1. Drop duplicate PMIDs with conflicting labels ---
    label_counts = df.groupby("pmid")["label"].nunique()
    conflicting_pmids = label_counts[label_counts > 1].index
    n_conflicting = df["pmid"].isin(conflicting_pmids).sum()
    df = df[~df["pmid"].isin(conflicting_pmids)].copy()

    # (defensive) drop any remaining exact-duplicate PMIDs, keep first
    n_exact_dupes = df["pmid"].duplicated().sum()
    df = df.drop_duplicates(subset="pmid", keep="first")

    # --- 2. Drop truncated/corrupted abstracts ---
    abs_len = df["abstract"].str.len()
    n_short = (abs_len < MIN_ABSTRACT_LEN).sum()
    df = df[abs_len >= MIN_ABSTRACT_LEN].copy()

    # --- 3. Missing titles/years: leave as NULL, just note counts ---
    n_missing_title = df["title"].isna().sum()
    n_missing_year = df["year"].isna().sum()

    # --- 4. Whitespace cleanup ---
    df["title"] = clean_whitespace(df["title"].fillna(""))
    df["title"] = df["title"].replace("", pd.NA)
    df["abstract"] = clean_whitespace(df["abstract"])
    df["journal"] = clean_whitespace(df["journal"])

    # Encode label as binary target for modeling convenience
    df["label_binary"] = (df["label"] == "category_b").astype(int)

    # --- Load into SQLite ---
    conn = sqlite3.connect(OUTPUT_DB)
    df.to_sql("abstracts", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pmid ON abstracts(pmid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON abstracts(label)")
    conn.commit()
    conn.close()

    # --- Summary report ---
    print("=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Rows in raw CSV:                    {n_start}")
    print(f"Dropped (conflicting-label PMIDs):  {n_conflicting} ({len(conflicting_pmids)} unique PMIDs)")
    print(f"Dropped (exact duplicate PMIDs):    {n_exact_dupes}")
    print(f"Dropped (abstract < {MIN_ABSTRACT_LEN} chars):        {n_short}")
    print(f"Rows remaining:                      {len(df)}")
    print(f"  -> missing title:                  {n_missing_title}")
    print(f"  -> missing year:                   {n_missing_year}")
    print()
    print("Final label distribution:")
    print(df["label"].value_counts().to_string())
    print()
    print(f"Saved to {OUTPUT_DB}, table 'abstracts'")

if __name__ == "__main__":
    main()
