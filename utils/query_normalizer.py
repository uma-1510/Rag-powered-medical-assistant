"""
Symptom normalization layer.

Maps plain-English user language
to medically relevant retrieval terms.
"""

import re


# ---- synonym expansion dictionary ----
SYMPTOM_MAP = {
    "sore throat": [
        "pharyngitis",
        "throat infection",
        "throat inflammation"
    ],

    "cannot see clearly": [
        "blurred vision",
        "vision loss",
        "eye disorder"
    ],

    "fell": [
        "injury",
        "trauma",
        "wound",
        "abrasion"
    ],

    "cut": [
        "laceration",
        "wound care",
        "bleeding injury"
    ],

    "fever": [
        "infection",
        "high temperature",
        "viral illness"
    ],

    "headache": [
        "migraine",
        "neurological pain",
        "head pain"
    ],

    "cough": [
        "respiratory infection",
        "bronchitis",
        "lung infection"
    ]
}


def normalize_query(query: str) -> str:
    """
    Expand user query with medical terminology
    to improve semantic retrieval.
    """

    expanded_terms = []
    q_lower = query.lower()

    for phrase, medical_terms in SYMPTOM_MAP.items():
        if re.search(rf"\b{re.escape(phrase)}\b", q_lower):
            expanded_terms.extend(medical_terms)

    if expanded_terms:
        expanded_query = query + " " + " ".join(expanded_terms)
        return expanded_query

    return query
