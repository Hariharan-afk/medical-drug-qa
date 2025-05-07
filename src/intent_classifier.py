# # src/intent_classifier.py
# import os
# from typing import List, Dict
# from fuzzywuzzy import fuzz
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import spacy

# # === Load models === #
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# ner_model = spacy.load(os.path.join(os.path.dirname(__file__), "en_ner_bc5cdr_md"))

# # === Section Aliases for semantic section matching === #
# SECTION_ALIASES = {
#     "Description": [
#         "What is this drug?",
#         "What does this medicine do?",
#         "Basic information about the drug",
#         "Describe this drug",
#         "Give an overview of this drug",
#         "What is the drug?",
#         "I need the description of the drug",
#         "Give me the description of the drug"
#     ],
#     "Before Using": [
#         "Things to tell your doctor before using this drug",
#         "Can I take this drug with another?",
#         "Warnings and interactions",
#         "Will this drug interact with another drug?",
#         "Can drug 1 and drug 2 be taken together?",
#         "Can I take this drug while having a specific medical condition?",
#         "When should I avoid taking this drug?"
#     ],
#     "Proper Use": [
#         "How to take this medication",
#         "Instructions for using the drug",
#         "What should I do if I miss a dose?",
#         "Dosage and timing",
#         "How to store this drug?",
#         "Can I store this drug outside?",
#         "Can this drug be stored at room temperature?",
#         "How should I use this drug to treat a condition?"
#     ],
#     "Precautions": [
#         "Things to be careful about",
#         "Risks while using the drug",
#         "Conditions that affect drug safety",
#         "What kind of allergic reactions can this drug produce?",
#         "Does this drug cause allergies?",
#         "Can this drug cause a specific symptom?"
#     ],
#     "Side Effects": [
#         "What are the side effects of using this drug?",
#         "Symptoms or reactions from the drug",
#         "Adverse effects",
#         "What are the common side effects of this drug?",
#         "What are the rare or less common side effects of this drug?"
#     ]
# }

# def extract_drug_entities(text: str) -> List[str]:
#     doc = ner_model(text)
#     return [ent.text.lower() for ent in doc.ents if ent.label_ == "CHEMICAL"]

# def match_section_semantically(question: str, section_map: Dict[str, List[str]]) -> str:
#     question_emb = embedding_model.encode([question], normalize_embeddings=True)

#     all_phrases = []
#     section_lookup = []

#     for section, phrases in section_map.items():
#         for p in phrases:
#             all_phrases.append(p)
#             section_lookup.append(section)

#     phrase_embs = embedding_model.encode(all_phrases, normalize_embeddings=True)
#     scores = np.dot(phrase_embs, question_emb.T).squeeze()
#     best_section = section_lookup[int(np.argmax(scores))]
#     return best_section

# def extract_intents(
#     question: str,
#     drug_names: List[str],
#     section_names: List[str]  # Retained for compatibility
# ) -> Dict:
#     question_lower = question.lower()
#     intents = {"drug_list": []}

#     # Step 1: NER
#     ner_drugs = extract_drug_entities(question)

#     for ner_drug in ner_drugs:
#         matches = [d for d in drug_names if ner_drug in d.lower()]
#         intents["drug_list"].extend(matches)

#     if intents["drug_list"]:
#         intents["drug"] = intents["drug_list"][0]

#     # Step 2: Fuzzy fallback if NER fails or yields nothing usable
#     if not intents.get("drug_list"):
#         best_match = None
#         best_score = 0

#         for d in drug_names:
#             score = fuzz.token_sort_ratio(d.lower(), question_lower)
#             if score > best_score:
#                 best_score = score
#                 best_match = d

#         if best_score > 70:
#             intents["drug_list"] = [best_match]
#             intents["drug"] = best_match

#     # Step 3: Semantic section classification
#     section = match_section_semantically(question, SECTION_ALIASES)
#     intents["section"] = section
#     intents["section_list"] = [section]  # Ensure consistency for downstream processing

#     return intents



# import os
# from typing import List, Dict
# from fuzzywuzzy import fuzz
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline

# # === Load models === #
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize BioBERT NER pipeline
# biobert_ner_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# biobert_ner_model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
# ner_model = pipeline("ner", model=biobert_ner_model, tokenizer=biobert_ner_tokenizer, aggregation_strategy="simple")

# # === Section Aliases for semantic section matching === #
# SECTION_ALIASES = {
#     "Description": [
#         "What is this drug?",
#         "What does this medicine do?",
#         "Basic information about the drug",
#         "Describe this drug",
#         "Give an overview of this drug",
#         "What is the drug?",
#         "I need the description of the drug",
#         "Give me the description of the drug"
#     ],
#     "Before Using": [
#         "Things to tell your doctor before using this drug",
#         "Can I take this drug with another?",
#         "Warnings and interactions",
#         "Will this drug interact with another drug?",
#         "Can drug 1 and drug 2 be taken together?",
#         "Can I take this drug while having a specific medical condition?",
#         "When should I avoid taking this drug?"
#     ],
#     "Proper Use": [
#         "How to take this medication",
#         "Instructions for using the drug",
#         "What should I do if I miss a dose?",
#         "Dosage and timing",
#         "How to store this drug?",
#         "Can I store this drug outside?",
#         "Can this drug be stored at room temperature?",
#         "How should I use this drug to treat a condition?"
#     ],
#     "Precautions": [
#         "Things to be careful about",
#         "Risks while using the drug",
#         "Conditions that affect drug safety",
#         "What kind of allergic reactions can this drug produce?",
#         "Does this drug cause allergies?",
#         "Can this drug cause a specific symptom?"
#     ],
#     "Side Effects": [
#         "What are the side effects of using this drug?",
#         "Symptoms or reactions from the drug",
#         "Adverse effects",
#         "What are the common side effects of this drug?",
#         "What are the rare or less common side effects of this drug?"
#     ]
# }

# def extract_drug_entities(text: str) -> List[str]:
#     """Extract drug/chemical entities using BioBERT NER"""
#     entities = ner_model(text)
#     # Filter for chemical/drug entities (entity_group will depend on the model's labels)
#     chemical_entities = [
#         ent["word"].lower() 
#         for ent in entities 
#         if ent["entity_group"] in ["CHEMICAL", "DRUG"]  # Adjust based on model's actual labels
#     ]
#     return chemical_entities

# def match_section_semantically(question: str, section_map: Dict[str, List[str]]) -> str:
#     question_emb = embedding_model.encode([question], normalize_embeddings=True)

#     all_phrases = []
#     section_lookup = []

#     for section, phrases in section_map.items():
#         for p in phrases:
#             all_phrases.append(p)
#             section_lookup.append(section)

#     phrase_embs = embedding_model.encode(all_phrases, normalize_embeddings=True)
#     scores = np.dot(phrase_embs, question_emb.T).squeeze()
#     best_section = section_lookup[int(np.argmax(scores))]
#     return best_section

# def extract_intents(
#     question: str,
#     drug_names: List[str],
#     section_names: List[str]  # Retained for compatibility
# ) -> Dict:
#     question_lower = question.lower()
#     intents = {"drug_list": []}

#     # Step 1: NER with BioBERT
#     ner_drugs = extract_drug_entities(question)

#     for ner_drug in ner_drugs:
#         # Improved matching to handle partial matches
#         matches = [d for d in drug_names if ner_drug in d.lower() or d.lower() in ner_drug]
#         intents["drug_list"].extend(matches)

#     # Remove duplicates while preserving order
#     intents["drug_list"] = list(dict.fromkeys(intents["drug_list"]))
    
#     if intents["drug_list"]:
#         intents["drug"] = intents["drug_list"][0]

#     # Step 2: Fuzzy fallback if NER fails or yields nothing usable
#     if not intents.get("drug_list"):
#         best_match = None
#         best_score = 0

#         for d in drug_names:
#             score = fuzz.token_sort_ratio(d.lower(), question_lower)
#             if score > best_score:
#                 best_score = score
#                 best_match = d

#         if best_score > 70:
#             intents["drug_list"] = [best_match]
#             intents["drug"] = best_match

#     # Step 3: Semantic section classification
#     section = match_section_semantically(question, SECTION_ALIASES)
#     intents["section"] = section
#     intents["section_list"] = [section]  # Ensure consistency for downstream processing

#     return intents


# src/intent_classifier.py

# import numpy as np
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# from sentence_transformers import SentenceTransformer

# # === Your section aliases ===
# SECTION_ALIASES = {
#     "Description": [
#         "What is this drug?",
#         "What does this medicine do?",
#         "Basic information about the drug",
#         "Describe this drug",
#         "Give an overview of this drug",
#         "What is the drug?",
#         "I need the description of the drug",
#         "Give me the description of the drug"
#     ],
#     "Before Using": [
#         "Things to tell your doctor before using this drug",
#         "Can I take this drug with another?",
#         "Warnings and interactions",
#         "Will this drug interact with another drug?",
#         "Can drug 1 and drug 2 be taken together?",
#         "Can I take this drug while having a specific medical condition?",
#         "When should I avoid taking this drug?"
#     ],
#     "Proper Use": [
#         "How to take this medication",
#         "Instructions for using the drug",
#         "What should I do if I miss a dose?",
#         "Dosage and timing",
#         "How to store this drug?",
#         "Can I store this drug outside?",
#         "Can this drug be stored at room temperature?",
#         "How should I use this drug to treat a condition?"
#     ],
#     "Precautions": [
#         "Things to be careful about",
#         "Risks while using the drug",
#         "Conditions that affect drug safety",
#         "What kind of allergic reactions can this drug produce?",
#         "Does this drug cause allergies?",
#         "Can this drug cause a specific symptom?"
#     ],
#     "Side Effects": [
#         "What are the side effects of using this drug?",
#         "Symptoms or reactions from the drug",
#         "Adverse effects",
#         "What are the common side effects of this drug?",
#         "What are the rare or less common side effects of this drug?"
#     ]
# }

# # === Load NER (BioBERT) ===
# tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
# ner_model = AutoModelForTokenClassification.from_pretrained(
#     "dmis-lab/biobert-base-cased-v1.1",
#     ignore_mismatched_sizes=True
# )
# ner_pipeline = pipeline(
#     "ner",
#     model=ner_model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple"  # merges subwords into full entities
# )

# # === Load semantic embedder (MiniLM‐v6) ===
# semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Precompute normalized alias embeddings for each section
# _alias_embeddings = {}
# for section, examples in SECTION_ALIASES.items():
#     emb = semantic_model.encode(examples, convert_to_numpy=True, show_progress_bar=False)
#     emb /= np.linalg.norm(emb, axis=1, keepdims=True)
#     _alias_embeddings[section] = emb

# def extract_drug_name(query: str) -> str | None:
#     """
#     Run NER on query and return the longest CHEMICAL/DRUG entity, if any.
#     """
#     entities = ner_pipeline(query)
#     # keep only chemical/drug labels
#     drug_entities = [
#         ent["word"]
#         for ent in entities
#         if ent["entity_group"].lower() in ("chemical", "drug")
#     ]
#     return max(drug_entities, key=len) if drug_entities else None

# def classify_section(query: str, threshold: float = 0.3) -> str:
#     """
#     Embed the query and compute max cosine similarity against each section's aliases.
#     Returns the best section (even if below threshold).
#     """
#     q_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
#     q_emb /= np.linalg.norm(q_emb)
#     best_sec, best_score = None, -1.0

#     for section, emb in _alias_embeddings.items():
#         sims = emb @ q_emb  # inner products on normalized vectors = cosine
#         score = sims.max()
#         if score > best_score:
#             best_score, best_sec = score, section

#     return best_sec

# def classify_intent(query: str) -> dict:
#     """
#     Returns a dict with:
#       - drug_name: extracted string or None
#       - section: one of SECTION_ALIASES keys
#     """
#     return {
#         "drug_name": extract_drug_name(query),
#         "section": classify_section(query)
#     }

# if __name__ == "__main__":
#     # quick smoke test
#     sample = "What are the side effects of Ibuprofen?"
#     intent = classify_intent(sample)
#     print(intent)


# src/intent_classifier.py

import re
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer

# ─── 1. Build the drug dictionary ────────────────────────────────────────────────

_DRUG_ROUTE_PATTERN = re.compile(r"\s*\([^)]*route\)", flags=re.IGNORECASE)

def load_drug_dict(csv_path: str) -> Dict[str, str]:
    """
    Reads the CSV, extracts unique drug names (dropping any '(… route)' suffix),
    and returns a dict mapping lowercase name -> canonical name.
    """
    df = pd.read_csv(csv_path, usecols=["drug_name"])
    unique_names = df["drug_name"].unique()
    
    clean_names = set()
    for nm in unique_names:
        clean = _DRUG_ROUTE_PATTERN.sub("", nm).strip()
        clean_names.add(clean)
    
    # lowercase lookup -> canonical form
    return {name.lower(): name for name in clean_names}

def extract_drug_from_query(query: str, drug_dict: Dict[str, str]) -> Optional[str]:
    """
    Scans the query for any drug in drug_dict.
    Returns the canonical drug name if found, else None.
    """
    q = query.lower()
    # sort by length desc so "ibuprofen lysine" matches before "ibuprofen"
    for name in sorted(drug_dict.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", q):
            return drug_dict[name]
    return None

# load once at startup
DRUG_DICT = load_drug_dict("data/flattened_drug_dataset_cleaned.csv")


# ─── 2. Section Aliases + Semantic Embeddings ────────────────────────────────────

SECTION_ALIASES = {
    "Description": [
        "What is this drug?",
        "What does this medicine do?",
        "Basic information about the drug",
        "Describe this drug",
        "Give an overview of this drug",
        "What is the drug?",
        "I need the description of the drug",
        "Give me the description of the drug"
    ],
    "Before Using": [
        "Things to tell your doctor before using this drug",
        "Can I take this drug with another?",
        "Warnings and interactions",
        "Will this drug interact with another drug?",
        "Can drug 1 and drug 2 be taken together?",
        "Can I take this drug while having a specific medical condition?",
        "When should I avoid taking this drug?"
    ],
    "Proper Use": [
        "How to take this medication",
        "Instructions for using the drug",
        "What should I do if I miss a dose?",
        "Dosage and timing",
        "How to store this drug?",
        "Can I store this drug outside?",
        "Can this drug be stored at room temperature?",
        "How should I use this drug to treat a condition?"
    ],
    "Precautions": [
        "Things to be careful about",
        "Risks while using the drug",
        "Conditions that affect drug safety",
        "What kind of allergic reactions can this drug produce?",
        "Does this drug cause allergies?",
        "Can this drug cause a specific symptom?"
    ],
    "Side Effects": [
        "What are the side effects of using this drug?",
        "Symptoms or reactions from the drug",
        "Adverse effects",
        "What are the common side effects of this drug?",
        "What are the rare or less common side effects of this drug?"
    ]
}

# load and precompute normalized embeddings
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
_alias_embeddings: Dict[str, np.ndarray] = {}
for section, examples in SECTION_ALIASES.items():
    emb = semantic_model.encode(examples, convert_to_numpy=True, show_progress_bar=False)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    _alias_embeddings[section] = emb


# ─── 3. Intent Classification ───────────────────────────────────────────────────

def extract_drug_name(query: str) -> Optional[str]:
    """
    Look up the drug in our prebuilt dictionary.
    Returns the canonical drug name, or None if not found.
    """
    drug = extract_drug_from_query(query, DRUG_DICT)
    if not drug:
        # you can log or notify here if you want:
        # print(f"No info on '{query}' in our drug dictionary.")
        return None
    return drug

def classify_section(query: str) -> str:
    """
    Embed the query and pick the section whose aliases have the highest cosine similarity.
    """
    q_emb = semantic_model.encode([query], convert_to_numpy=True)[0]
    q_emb /= np.linalg.norm(q_emb)
    
    best_sec, best_score = None, -1.0
    for section, emb in _alias_embeddings.items():
        score = float((emb @ q_emb).max())
        if score > best_score:
            best_score, best_sec = score, section
    
    return best_sec

def classify_intent(query: str) -> dict:
    """
    Returns:
      - drug_name: the matched drug or None
      - section:   the best‐matching section
    """
    return {
        "drug_name": extract_drug_name(query),
        "section":   classify_section(query)
    }

# ─── 4. Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for q in [
        "What are the side effects of Ibuprofen Lysine?",
        "How should I store Azithromycin?",
        "Tell me about Paracetamol"
    ]:
        print(q, "→", classify_intent(q))

