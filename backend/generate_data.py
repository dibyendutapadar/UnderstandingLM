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
import math
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
    """An object phrase.

    Two complementary color behaviors that, together, let the color<->fruit
    analogy work even in 3D:
      * fruits carry their FIXED signature color (red apple, yellow banana ...)
        -> the consistent offset that powers `apple - red + yellow ~= banana`
      * animals carry a RANDOM color -> every color shares an
        'adjective-before-a-noun' context, so the colors form their own cluster
        (otherwise each color would just collapse onto its single fruit).
    """
    obj = random.choice(OBJECTS)
    if obj in COLOR_FRUIT and random.random() < 0.85:
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


def t_color_fruit_adj():
    """Color directly adjacent to its fruit so the association is robust to
    subsampling: 'he like the red apple .'"""
    fruit = random.choice(FRUITS)
    pro = random.choice(["he", "she"] + SUBJECT_NOUNS)
    det = [] if pro in ("he", "she") else ["the"]
    return det + [pro, random.choice(VERBS), "the", COLOR_FRUIT[fruit], fruit, "."]


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
    (t_color_fruit_adj, 12),  # analogy-critical (adjacency robust to subsampling)
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
# OpenAI generation — short STORIES (richer, more natural co-occurrence than
# isolated templated sentences, which is what makes the embedding geometry good)
# ---------------------------------------------------------------------------
STORY_SYSTEM_PROMPT = """You write tiny stories in a restricted toy language.

You may ONLY use these exact tokens — no other words, no capital letters, no \
contractions, no numbers:

{vocab}

Formatting rules (follow EXACTLY):
- Write {per_call} separate little stories. Put ONE story per line.
- Each story is 4 to 8 short sentences flowing together on that single line.
- Lowercase only. Put a space between every token, INCLUDING punctuation.
- End every sentence with " ." or " ?". You may use " ," inside a sentence.
- No numbering, no titles, no commentary — only the story lines.

Language rules:
- Keep sentences short and grammatical for this toy language (subject verb \
object, "the X is ADJ", etc.).
- Vary the people, verbs, objects, colors, places, and sentiment across stories.
- Respect these fixed associations so the world is consistent:
    apple is red , banana is yellow , grape is green , rice is blue ;
    he goes with man / boy / king ; she goes with woman / girl / queen .

Example of ONE story line:
the boy is happy . he see the red apple in the park . the girl give the apple \
to him ? she is very good . the king and the queen are happy , the boy eat the \
apple .
"""


def _clean_line(line):
    return " ".join(line.strip().lower().split())


def generate_openai_stories(num_stories, model, per_call=10, workers=8, seed=0):
    """Generate `num_stories` story lines via the OpenAI API, in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    client = OpenAI()  # reads OPENAI_API_KEY from env
    system = STORY_SYSTEM_PROMPT.format(vocab="  ".join(V.VOCAB), per_call=per_call)

    n_calls = math.ceil(num_stories / per_call)

    def one_call(i):
        resp = client.chat.completions.create(
            model=model,
            temperature=1.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"Write {per_call} new and varied stories."},
            ],
        )
        text = resp.choices[0].message.content or ""
        return [_clean_line(ln) for ln in text.splitlines() if _clean_line(ln)]

    stories, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one_call, i) for i in range(n_calls)]
        for fut in as_completed(futures):
            try:
                stories.extend(fut.result())
            except Exception as e:  # noqa: BLE001 — keep going, report at end
                print(f"  ! call failed: {e}")
            done += 1
            print(f"  ... {done}/{n_calls} calls, {len(stories)} story lines")
    return stories


def main():
    ap = argparse.ArgumentParser(description="Generate candidate corpus text.")
    ap.add_argument("--count", type=int, default=2000,
                    help="number of stories (OpenAI) or sentences (--dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="use Python sentence templates instead of the OpenAI API")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--per-call", type=int, default=10, help="stories per API call")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(here, "data", "raw_sentences.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.dry_run:
        print(f"Generating {args.count} template sentences (--dry-run) ...")
        lines = generate_templates(args.count, seed=args.seed)
        unit = "sentences"
    else:
        print(f"Generating ~{args.count} stories via OpenAI ({args.model}) ...")
        lines = generate_openai_stories(
            args.count, args.model, args.per_call, args.workers, args.seed
        )
        unit = "story lines"

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} {unit} -> {out_path}")
    print("Next: python validate.py  (splits stories into sentences + checks vocab)")


if __name__ == "__main__":
    main()
