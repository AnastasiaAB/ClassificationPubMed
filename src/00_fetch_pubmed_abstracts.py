"""
00_fetch_pubmed_abstracts.py

Pulls PubMed abstracts for two MeSH categories via NCBI's E-utilities API
and saves them to a single labeled CSV, ready to load into SQLite/Pandas
in the next project step.

Usage:
    python 00_fetch_pubmed_abstracts.py

Notes:
- No API key is required for light use (~3 requests/second limit).
  If you get an NCBI API key later, set NCBI_API_KEY below to raise
  the rate limit to 10 requests/second.
- NCBI asks that you set a real email/tool name so they can contact you
  if something in your usage looks off. Fill those in below.
"""

import time
import csv
import xml.etree.ElementTree as ET
import requests
import os
from dotenv import load_dotenv
from config import RAW_CSV, ROOT

# ---- Configuration ----------------------------------------------------
load_dotenv(ROOT / ".env")

NCBI_EMAIL = os.environ.get("NCBI_EMAIL")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")

NCBI_TOOL = "ds-portfolio-project"
NCBI_API_KEY = None                 # optional: set to your key string to speed up requests
 
# The two MeSH categories to classify between. Swap these for any two
# distinct topics you'd rather work with.
CATEGORY_A = "Neoplasms[MeSH Major Topic]"
CATEGORY_B = "Cardiovascular Diseases[MeSH Major Topic]"
 
N_PER_CATEGORY = 1500                # abstracts to pull per category
OUTPUT_CSV = RAW_CSV
 
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_DELAY = 0.34                 # ~3 req/sec without an API key

# ---- Functions ----------------------------------------------------------
def element_text(el):
    return "".join(el.itertext()).strip() if el is not None else ""

def esearch_pmids(query: str, retmax: int) -> list[str]:
    """Return a list of PubMed IDs matching a MeSH query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "email": NCBI_EMAIL,
        "tool": NCBI_TOOL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
 
    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]
 
 
def efetch_records(pmids: list[str]) -> list[dict]:
    """Fetch title/abstract/journal/year for a batch of PMIDs."""
    if not pmids:
        return []
 
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": NCBI_EMAIL,
        "tool": NCBI_TOOL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
 
    resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
 
    root = ET.fromstring(resp.content)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_els = article.findall(".//AbstractText")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//PubDate/Year")

        title = element_text(title_el)
        abstract_text = " ".join(element_text(el) for el in abstract_els).strip()
 
        # Skip records with no real abstract text -- not usable for
        # a text classification task.
        if not abstract_text:
            continue
 
        records.append({
            "pmid": pmid_el.text if pmid_el is not None else "",
            "title": title_el.text if title_el is not None else "",
            "abstract": abstract_text,
            "journal": journal_el.text if journal_el is not None else "",
            "year": year_el.text if year_el is not None else "",
        })
 
    return records
 
 
def fetch_category(query: str, label: str, n: int) -> list[dict]:
    """Fetch n labeled records for a single MeSH query, batching efetch calls."""
    print(f"Searching PubMed for: {query}")
    pmids = esearch_pmids(query, retmax=n)
    print(f"  Found {len(pmids)} candidate PMIDs")
 
    all_records = []
    batch_size = 200  # efetch handles batches of PMIDs well
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        records = efetch_records(batch)
        for r in records:
            r["label"] = label
        all_records.extend(records)
        print(f"  Fetched {len(all_records)}/{len(pmids)} abstracts so far...")
        time.sleep(REQUEST_DELAY)
 
    return all_records

def main():
    if not NCBI_EMAIL:
        raise SystemExit("Set NCBI_EMAIL in a .env file (copy .env.example). "
                         "NCBI asks for a contact address.")
    category_a_records = fetch_category(CATEGORY_A, "category_a", N_PER_CATEGORY)
    time.sleep(REQUEST_DELAY)
    category_b_records = fetch_category(CATEGORY_B, "category_b", N_PER_CATEGORY)
 
    all_records = category_a_records + category_b_records
    print(f"\nTotal usable abstracts: {len(all_records)}")
    print(f"  {CATEGORY_A}: {len(category_a_records)}")
    print(f"  {CATEGORY_B}: {len(category_b_records)}")
 
    fieldnames = ["pmid", "title", "abstract", "journal", "year", "label"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
 
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

