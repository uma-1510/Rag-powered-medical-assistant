MERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "unconsious",
    "severe bleeding",
    "vision loss",
    "head injury",
    "vomitings",
    "diaherra",
    "more than 5 days"
]


def detect_emergency(query, retrieved_chunks):

    q = query.lower()

    if any(term in q for term in MERGENCY_KEYWORDS):
        return True

    for chunk in retrieved_chunks:
        if chunk.get("is_emergency"):
            return True

    return False
