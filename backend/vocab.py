"""The 50-token synthetic language.

This module is the single source of truth for the vocabulary. It defines the
tokens, their categories, and the structural relationships (gender pairs,
color<->fruit pairings) that the corpus grammar leans on to make analogies come
out *linear* in the learned 3D embedding space.

Running this file writes ../shared/vocab.json so the frontend can import the
exact same vocabulary the backend trains on.
"""

import json
import os

# ---------------------------------------------------------------------------
# Categories -> tokens.  Order here defines token ids (flattened, top to bottom).
# 50 tokens total = 47 words + 3 punctuation.
# ---------------------------------------------------------------------------
CATEGORIES = {
    # gendered pairs -> drive the "gender axis"
    "people": ["he", "she", "man", "woman", "boy", "girl", "king", "queen"],
    "fruit": ["apple", "banana", "grape", "rice"],
    "color": ["red", "yellow", "green", "blue"],
    "animal": ["cat", "dog", "fish", "bird"],
    "sentiment": ["happy", "sad", "good", "bad"],
    "size": ["big", "small"],
    "verb": ["like", "hate", "eat", "see", "want", "give"],
    "function": ["is", "are", "the", "a", "and", "very", "not"],
    "place": ["in", "on", "with", "park", "house", "school"],
    "time": ["today", "tomorrow"],
    "punctuation": [".", ",", "?"],
}

# Flattened vocabulary (token id = index in this list).
VOCAB = [tok for toks in CATEGORIES.values() for tok in toks]

word2idx = {w: i for i, w in enumerate(VOCAB)}
idx2word = {i: w for i, w in enumerate(VOCAB)}

# token -> category (reverse map)
token2category = {tok: cat for cat, toks in CATEGORIES.items() for tok in toks}

# ---------------------------------------------------------------------------
# Structural relationships used by the grammar (Step 2) and verified as
# analogies after training (Step 3).
# ---------------------------------------------------------------------------

# Male/female counterparts. The grammar co-locates each male word with male
# partners and each female word with female partners so a single, consistent
# "gender" offset vector emerges:  king - man + woman ~= queen.
GENDER_PAIRS = [
    ("he", "she"),
    ("man", "woman"),
    ("boy", "girl"),
    ("king", "queen"),
]
MALE = [m for m, _ in GENDER_PAIRS]
FEMALE = [f for _, f in GENDER_PAIRS]

# Each fruit gets a signature color via fixed "<color> <fruit>" pairings. The
# constant color offset makes:  apple - red + yellow ~= banana.
COLOR_FRUIT = {
    "apple": "red",
    "banana": "yellow",
    "grape": "green",
    "rice": "blue",
}

# Analogy probes checked automatically by train_embeddings.py.
# (positive terms) - (negative term) ~= expected
ANALOGIES = [
    {"a": "king", "b": "man", "c": "woman", "expect": "queen"},
    {"a": "he", "b": "boy", "c": "girl", "expect": "she"},
    {"a": "apple", "b": "red", "c": "yellow", "expect": "banana"},
]


def is_punctuation(tok: str) -> bool:
    return token2category.get(tok) == "punctuation"


def export(path: str | None = None) -> str:
    """Write shared/vocab.json (the frontend/backend contract). Returns the path."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "shared", "vocab.json")
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "vocab": VOCAB,
        "size": len(VOCAB),
        "categories": CATEGORIES,
        "token2category": token2category,
        "gender_pairs": GENDER_PAIRS,
        "color_fruit": COLOR_FRUIT,
        "analogies": ANALOGIES,
        "counts": {
            "total": len(VOCAB),
            "words": sum(1 for t in VOCAB if not is_punctuation(t)),
            "punctuation": sum(1 for t in VOCAB if is_punctuation(t)),
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


if __name__ == "__main__":
    assert len(VOCAB) == 50, f"expected 50 tokens, got {len(VOCAB)}"
    assert len(set(VOCAB)) == 50, "duplicate token in VOCAB"
    out = export()
    print(f"vocab size: {len(VOCAB)} (47 words + 3 punctuation expected)")
    print(f"wrote {out}")
