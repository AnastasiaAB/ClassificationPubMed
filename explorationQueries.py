# %%
import sqlite3
import pandas as pd

# %%
conn = sqlite3.connect("pubmed.db")
# %%
#Class balance check
query = """
SELECT label, COUNT(*) AS n
FROM abstracts
GROUP BY label
"""
# %%
df = pd.read_sql_query(query, conn)
df
# %%
#Abstracts per year, per label (spot data drift / era imbalance)
query = """
SELECT year, label, COUNT(*) AS n
FROM abstracts
WHERE year IS NOT NULL
GROUP BY year, label
ORDER BY year;
"""
# %%
df = pd.read_sql_query(query, conn)
df
# %%
#Top 10 journals by article count
query = """
SELECT journal, COUNT(*) AS n
FROM abstracts
GROUP BY journal
ORDER BY n DESC
LIMIT 10
"""
# %%
df = pd.read_sql_query(query, conn)
df
# %%
"""Window function: rank journals within each label by article count
-- (useful for spotting whether one journal dominates a class -- a
--  potential source of leakage/bias worth flagging in your writeup)"""
query = """
SELECT label, journal, n,
       RANK() OVER (PARTITION BY label ORDER BY n DESC) AS rank_in_label
FROM (
    SELECT label, journal, COUNT(*) AS n
    FROM abstracts
    GROUP BY label, journal
)
QUALIFY rank_in_label <= 5;
"""
# %%
df = pd.read_sql_query(query, conn)
df
# %%
