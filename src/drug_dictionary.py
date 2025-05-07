# src/drug_dictionary.py

import re
import pandas as pd
from typing import Dict, Optional

_DRUG_ROUTE_PATTERN = re.compile(r"\s*\([^)]*route\)", flags=re.IGNORECASE)

def load_drug_dict(csv_path: str) -> Dict[str, str]:
    """
    Reads the CSV, extracts unique drug names (dropping any '(… route)' suffix),
    and returns a dict mapping lowercase name -> canonical name.
    """
    df = pd.read_csv(csv_path, usecols=["drug_name"])
    # remove duplicates
    names = df["drug_name"].unique()
    clean_names = set()
    for nm in names:
        # strip off " (oral route)", " (intravenous route)", etc.
        clean = _DRUG_ROUTE_PATTERN.sub("", nm).strip()
        clean_names.add(clean)

    # build lookup dict
    # lowercase key -> cleaned display name
    return {name.lower(): name for name in clean_names}

def extract_drug_from_query(query: str, drug_dict: Dict[str, str]) -> Optional[str]:
    """
    Scans the query for any drug in drug_dict.
    Returns the canonical drug name if found, else None.
    We match longest names first to avoid partial matches.
    """
    q = query.lower()
    # sort keys by length descending so that e.g. "ibuprofen lysine" matches before "ibuprofen"
    for name in sorted(drug_dict.keys(), key=len, reverse=True):
        # word-boundary match
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, q):
            return drug_dict[name]
    return None

if __name__ == "__main__":
    # quick test
    drug_dict = load_drug_dict("data/flattened_drug_dataset_cleaned.csv")
    for q in [
        "What are the side effects of Ibuprofen lysine?",
        "Can I take azithromycin with ibuprofen?",
        "Tell me about Paracetamol"
        "What is Abobotulinumtoxina?"
    ]:
        found = extract_drug_from_query(q, drug_dict)
        print(f"Query: {q!r}  →  Drug: {found}")
