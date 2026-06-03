"""Step 2 — generate candidate sentences for the synthetic language.

Two modes:

  --dry-run    Generate from Python grammar templates. No API, no spend.
               Guarantees the analogy-supporting co-occurrence structure and
               volume. Use this for testing.

  (default)    Use the OpenAI API. The prompt hands the model the 50-token
               vocabulary + the grammar patterns and asks it to emit sentences
               using ONLY those tokens. Gives more natural variety; the checker
               (validate.py) still enforces the hard vocabulary constraint.

Output: writes raw candidate sentences (one per line) to data/raw_sentences.txt.
Run validate.py next to filter + build sentences.jsonl + corpus_stats.json.

The templates here are deliberately weighted toward the two relationships the
3D analogies depend on:
  * gender tagging   he <-> {king, man, boy},  she <-> {queen, woman, girl}
  * color<->fruit    fixed "<signature-color> <fruit>" pairings
"""

import argparse
import os
import random

import vocab as V

# ---------------------------------------------------------------------------
# Slot pools (all drawn from the 50-token vocab)
# ---------------------------------------------------------------------------
MALE_NOUNS = ["man", "boy", "king"]
FEMALE_NOUNS = ["woman", "girl", "queen"]
PEOPLE = V.CATEGORIES["people"]                      # incl. he/she
SUBJECT_NOUNS = MALE_NOUNS + FEMALE_NOUNS            # nameable people (no pronoun)

VERBS = V.CATEGORIES["verb"]                         # like hate eat see want give
FRUITS = V.CATEGORIES["fruit"]                       # apple banana grape rice
ANIMALS = V.CATEGORIES["animal"]                     # cat dog fish bird
OBJECTS = FRUITS + ANIMALS
COLORS = V.CATEGORIES["color"]
POS = ["happy", "good"]
NEG = ["sad", "bad"]
SENTIMENT = POS + NEG
SIZES = ["big", "small"]
PLACES = V.CATEGORIES["place"][3:]                   # park house school
TIMES = V.CATEGORIES["time"]                         # today tomorrow

COLOR_FRUIT = V.COLOR_FRUIT                           # apple->red, banana->yellow, ...

# Light role-context to nudge royal / adult / child apart along one axis.
ROLE_PLACE = {
    "king": "house", "queen": "house",
    "man": "house", "woman": "house",
    "boy": "school", "girl": "school",
}
ROLE_SIZE = {
    "king": "big", "queen": "big",
    "man": "big", "woman": "big",
    "boy": "small", "girl": "small",
}


def _noun_with_det(noun):
    return ["the", noun]


def _object_phrase():
    """An object, fruits half the time carrying their signature color."""
    obj = random.choice(OBJECTS)
    if obj in COLOR_FRUIT and random.random() < 0.6:
        return ["the", COLOR_FRUIT[obj], obj]
    return ["the", obj]


# ---------------------------------------------------------------------------
# Templates. Each returns a list of tokens. Weights below favor the
# analogy-critical patterns (gender tagging, color<->fruit).
# ---------------------------------------------------------------------------

def t_svo():
    subj = random.choice(SUBJECT_NOUNS)
    return _noun_with_det(subj) + [random.choice(VERBS)] + _object_phrase() + ["."]


def t_pronoun_svo():
    pro = random.choice(["he", "she"])
    return [pro, random.choice(VERBS)] + _object_phrase() + ["."]


def t_gender_tag():
    """The analogy backbone: pronoun agrees with a same-gender noun."""
    if random.random() < 0.5:
        return ["he", "is", "a", random.choice(MALE_NOUNS), "."]
    return ["she", "is", "a", random.choice(FEMALE_NOUNS), "."]


def t_same_gender_pair():
    """Two same-gender nouns co-occur -> tightens each gender cluster."""
    if random.random() < 0.5:
        a, b = random.sample(MALE_NOUNS, 2)
    else:
        a, b = random.sample(FEMALE_NOUNS, 2)
    return ["the", a, "and", "the", b, random.choice(VERBS)] + _object_phrase() + ["."]


def t_color_fruit():
    """Strong, frequent color<->fruit pairing."""
    fruit = random.choice(FRUITS)
    color = COLOR_FRUIT[fruit]
    return ["the", fruit, "is", color, "."]


def t_sentiment():
    subj = random.choice(SUBJECT_NOUNS + ["he", "she"])
    det = [] if subj in ("he", "she") else ["the"]
    word = random.choice(SENTIMENT)
    mod = random.choice([[], ["very"], ["not"]])
    return det + [subj, "is"] + mod + [word, "."]


def t_role_context():
    """Light role separation: who is where / what size."""
    noun = random.choice(SUBJECT_NOUNS)
    if random.random() < 0.5:
        return ["the", noun, "is", "in", "the", ROLE_PLACE[noun], "."]
    return ["the", noun, "is", ROLE_SIZE[noun], "."]


def t_svo_place():
    subj = random.choice(SUBJECT_NOUNS)
    return (
        _noun_with_det(subj)
        + [random.choice(VERBS)]
        + _object_phrase()
        + ["in", "the", random.choice(PLACES), "."]
    )


def t_question():
    pro = random.choice(["he", "she"])
    return [pro, random.choice(VERBS), "the", random.choice(OBJECTS), "?"]


def t_time():
    subj = random.choice(SUBJECT_NOUNS)
    return _noun_with_det(subj) + [random.choice(VERBS)] + _object_phrase() + [
        random.choice(TIMES), "."
    ]


def t_plural_are():
    """Plural subject -> 'are' agreement (mirror of 'is')."""
    a, b = random.sample(SUBJECT_NOUNS, 2)
    word = random.choice(SENTIMENT)
    mod = random.choice([[], ["very"], ["not"]])
    return ["the", a, "and", "the", b, "are"] + mod + [word, "."]


def t_with():
    """Companion phrase -> exercises 'with'."""
    subj = random.choice(SUBJECT_NOUNS)
    other = random.choice(SUBJECT_NOUNS + ANIMALS)
    return ["the", subj, random.choice(VERBS)] + _object_phrase() + [
        "with", "the", other, "."
    ]


def t_on():
    """Location 'on' a place."""
    animal = random.choice(ANIMALS)
    return ["the", animal, "is", "on", "the", random.choice(PLACES), "."]


def t_comma_list():
    """A comma-separated object list -> exercises ','."""
    a, b, c = random.sample(OBJECTS, 3)
    pro = random.choice(["he", "she"])
    return [pro, random.choice(VERBS), "the", a, ",", "the", b, "and", "the", c, "."]


# (template, weight)
TEMPLATES = [
    (t_svo, 16),
    (t_pronoun_svo, 10),
    (t_gender_tag, 14),      # analogy-critical
    (t_same_gender_pair, 8),
    (t_color_fruit, 14),     # analogy-critical
    (t_sentiment, 10),
    (t_role_context, 8),
    (t_svo_place, 8),
    (t_question, 4),
    (t_time, 4),
    (t_plural_are, 6),
    (t_with, 5),
    (t_on, 4),
    (t_comma_list, 5),
]
_FUNCS = [f for f, _ in TEMPLATES]
_WEIGHTS = [w for _, w in TEMPLATES]


def generate_templates(count, seed=0):
    rng = random.Random(seed)
    random.seed(seed)  # templates use the module-level random
    out = []
    for _ in range(count):
        fn = rng.choices(_FUNCS, weights=_WEIGHTS, k=1)[0]
        out.append(" ".join(fn()))
    return out


# ---------------------------------------------------------------------------
# OpenAI generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You generate sentences in a tiny artificial language.

You may ONLY use these exact tokens (nothing else, no other words, no \
capitalization, no contractions):

{vocab}

Rules:
- Output one sentence per line. No numbering, no commentary.
- Lowercase only. Separate every token (including punctuation) with a space.
- End each sentence with " ." or " ?".
- Keep sentences short (3-8 tokens), simple, and grammatical for this language.
- Follow patterns like:
    the boy eat the red apple .
    he is a king .
    she is a queen .
    the apple is red .
    the king and the man see the dog .
    the girl is very happy .
    she want the banana ?
- Respect these fixed associations: apple->red, banana->yellow, grape->green, \
rice->blue ; he goes with man/boy/king ; she goes with woman/girl/queen .
"""


def generate_openai(count, model, batch=50, seed=0):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    client = OpenAI()  # reads OPENAI_API_KEY from env
    system = SYSTEM_PROMPT.format(vocab="  ".join(V.VOCAB))

    out = []
    while len(out) < count:
        n = min(batch, count - len(out))
        resp = client.chat.completions.create(
            model=model,
            temperature=1.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Generate {n} sentences."},
            ],
        )
        text = resp.choices[0].message.content or ""
        for line in text.splitlines():
            line = " ".join(line.strip().lower().split())
            if line:
                out.append(line)
        print(f"  ... {len(out)}/{count} candidate sentences")
    return out[:count]


def main():
    ap = argparse.ArgumentParser(description="Generate candidate sentences.")
    ap.add_argument("--count", type=int, default=15000, help="number of sentences")
    ap.add_argument("--dry-run", action="store_true",
                    help="use Python templates instead of the OpenAI API")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--batch", type=int, default=50, help="sentences per API call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(here, "data", "raw_sentences.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mode = "templates (--dry-run)" if args.dry_run else f"OpenAI ({args.model})"
    print(f"Generating {args.count} candidate sentences via {mode} ...")

    if args.dry_run:
        sentences = generate_templates(args.count, seed=args.seed)
    else:
        sentences = generate_openai(args.count, args.model, args.batch, args.seed)

    with open(out_path, "w") as f:
        f.write("\n".join(sentences) + "\n")
    print(f"Wrote {len(sentences)} candidates -> {out_path}")
    print("Next: python validate.py")


if __name__ == "__main__":
    main()
