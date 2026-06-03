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
import json
import math
import os
import random

import vocab as V
from validate import split_sentences, validate_sentence

# ---------------------------------------------------------------------------
# Slot pools (all drawn from the 50-token vocab)
# ---------------------------------------------------------------------------
MALE_NOUNS = ["man", "boy", "king"]
FEMALE_NOUNS = ["woman", "girl", "queen"]
PEOPLE = V.CATEGORIES["people"]                      # incl. he/she
SUBJECT_NOUNS = MALE_NOUNS + FEMALE_NOUNS            # nameable people (no pronoun)

FRUITS = V.CATEGORIES["fruit"]                       # apple banana grape rice
ANIMALS = V.CATEGORIES["animal"]                     # cat dog fish bird
OBJECTS = FRUITS + ANIMALS
COLORS = V.CATEGORIES["color"]
POS = ["happy", "good"]
NEG = ["sad", "bad"]
SENTIMENT = POS + NEG
SIZES = ["big", "small"]
THINGS = V.CATEGORIES.get("thing", [])               # water sky
TIMES = V.CATEGORIES["time"]                          # today tomorrow

# "place" mixes prepositions and locations; split them so templates stay grammatical.
PREPOSITIONS = {"in", "on", "with", "to"}
PLACES = [p for p in V.CATEGORIES["place"] if p not in PREPOSITIONS]   # house school
# Transitive verbs vs motion verbs (go/come take "to <place>", not an object).
MOTION_VERBS = [v for v in V.CATEGORIES["verb"] if v in ("go", "come")]
VERBS = [v for v in V.CATEGORIES["verb"] if v not in ("go", "come")]   # like hate eat see want give

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


def t_they():
    """Plural pronoun 'they' as subject."""
    if random.random() < 0.5:
        return ["they", "are", random.choice(SENTIMENT), "."]
    return ["they", random.choice(VERBS)] + _object_phrase() + ["."]


def t_thing_desc():
    """Describe a 'thing' (water/sky) with a color -> covers water, sky, white."""
    if not THINGS:
        return ["the", "sky", "is", "blue", "."]
    thing = random.choice(THINGS)
    color = "white" if (thing == "sky" and random.random() < 0.5) else "blue"
    return ["the", thing, "is", color, "."]


def t_motion():
    """Motion verb + 'to <place>' -> covers go, come, to, house, school."""
    pro = random.choice(["he", "she", "they"] + SUBJECT_NOUNS)
    det = [] if pro in ("he", "she", "they") else ["the"]
    verb = random.choice(MOTION_VERBS) if MOTION_VERBS else "go"
    return det + [pro, verb, "to", "the", random.choice(PLACES), "."]


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
    (t_they, 6),
    (t_thing_desc, 5),
    (t_motion, 6),
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
# OpenAI generation — short PASSAGES (richer, more natural co-occurrence than
# isolated templated sentences, which is what makes the embedding geometry good)
# ---------------------------------------------------------------------------
STORY_SYSTEM_PROMPT = """You write tiny stories / paragraphs passages in a restricted toy language.

Build sentences mostly from these core tokens — no capital letters, no \
contractions, no numbers:

{vocab}

You may ALSO use a few simple "concept" words ONLY for definitions and \
groupings: people , animal , fruit , food , color , thing , place . Avoid any \
other words.

Formatting rules (follow EXACTLY):
- Write {per_call} separate passages. Put ONE passage per line.
- Each passage is 4 to 8 short sentences flowing together on that single line.
- Mix three kinds of passage, roughly equally:
    1) little stories,
    2) plain descriptions,
    3) DEFINITION / CATEGORY passages that state facts and groupings.
- Lowercase only. Put a space between every token, INCLUDING punctuation.
- End every sentence with " ." or " ?". You may use " ," inside a sentence.
- No numbering, no titles, no commentary — only the passage lines.

Language rules:
- Keep sentences short and grammatical, and they must be as logical as possible (see story passage example)
- Vary the people, verbs, objects, colors, places, and sentiment.
- make the sentences logical within the span of words. make sure the spread is good all across.
- Respect these fixed associations so the world is consistent:
    apple is red , banana is yellow , grape is green , rice is white, sky is blue;
    he goes with man / boy / king ; she goes with woman / girl / queen .

Definition / category passage — write MANY of these, they teach the meanings:
king is a man . king is he . queen is a woman . queen is she . \
the apple is red . the banana is yellow . king and queen are people . \
man and woman and boy and girl are people . apple and banana and grape are \
fruit . apple and banana and rice are food . cat and dog and fish are animal .

Story passage example:
he is a boy. the boy is happy . he see the red apple. she is a girl. she is happy. the boy and girl are happy. she like red apple. he see red apple. he give red apple to girl. he is sad. she is sad.\
he is a king. king is a man. she is a queen. the king and the queen are happy. they eat banana. banana is yellow. banana good. they are happy.\
he is a boy. he go to school today. school good, he happy. he go to school tomorrow. school bad, he sad.
she is a girl. she see sky. sky is blue.
king is a man. a man is he. he like dog. he hare cat. cat and dog are animal.
"""


def _clean_line(line):
    return " ".join(line.strip().lower().split())


def _line_report(line, temperature):
    """Validate one generated passage against the vocab. Records full detail so
    the raw output can be inspected by hand later."""
    toks = line.split()
    sents = split_sentences(toks)
    n_valid_sent = sum(1 for s in sents if validate_sentence(s)[0])
    unknown = sorted({t for t in toks if t not in V.word2idx})
    return {
        "temperature": temperature,
        "text": line,
        "num_tokens": len(toks),
        "num_sentences": len(sents),
        "num_valid_sentences": n_valid_sent,
        "unknown": unknown,
        "valid": len(unknown) == 0,
    }


def generate_openai_stories(num_passages, model, per_call, workers, temperatures, seed=0):
    """Generate passages via the OpenAI API in parallel, cycling through the
    given temperatures. Returns a list of validated per-passage records."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    client = OpenAI()  # reads OPENAI_API_KEY from env
    system = STORY_SYSTEM_PROMPT.format(vocab="  ".join(V.VOCAB), per_call=per_call)

    n_calls = math.ceil(num_passages / per_call)

    def one_call(i):
        temp = temperatures[i % len(temperatures)]
        resp = client.chat.completions.create(
            model=model,
            temperature=temp,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"Write {per_call} new and varied passages."},
            ],
        )
        text = resp.choices[0].message.content or ""
        return [
            _line_report(_clean_line(ln), temp)
            for ln in text.splitlines()
            if _clean_line(ln)
        ]

    records, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one_call, i) for i in range(n_calls)]
        for fut in as_completed(futures):
            try:
                records.extend(fut.result())
            except Exception as e:  # noqa: BLE001 — keep going, report at end
                print(f"  ! call failed: {e}")
            done += 1
            ok = sum(1 for r in records if r["valid"])
            print(f"  ... {done}/{n_calls} calls, {len(records)} passages "
                  f"({ok} fully valid)")
    return records


def _summarize(records):
    """Print a per-temperature validity breakdown of the generated passages."""
    by_temp = {}
    for r in records:
        d = by_temp.setdefault(r["temperature"], {"n": 0, "valid": 0,
                                                   "sent": 0, "valid_sent": 0})
        d["n"] += 1
        d["valid"] += r["valid"]
        d["sent"] += r["num_sentences"]
        d["valid_sent"] += r["num_valid_sentences"]
    print("\nValidation summary (passages contain only allowed vocab?):")
    print(f"  {'temp':>5}  {'passages':>8}  {'fully valid':>12}  {'salvageable sentences':>22}")
    for temp in sorted(by_temp):
        d = by_temp[temp]
        vp = 100 * d["valid"] / d["n"] if d["n"] else 0
        vs = 100 * d["valid_sent"] / d["sent"] if d["sent"] else 0
        print(f"  {temp:>5}  {d['n']:>8}  {d['valid']:>5} ({vp:5.1f}%)  "
              f"{d['valid_sent']:>7}/{d['sent']:<7} ({vs:5.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Generate candidate corpus text.")
    ap.add_argument("--count", type=int, default=2000,
                    help="number of passages (OpenAI) or sentences (--dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="use Python sentence templates instead of the OpenAI API")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--per-call", type=int, default=15, help="passages per API call")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls")
    ap.add_argument("--temperatures", default="0.7,0.9,1.1,1.3",
                    help="comma-separated temperatures to cycle through")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    out_path = args.out or os.path.join(data_dir, "raw_sentences.txt")
    os.makedirs(data_dir, exist_ok=True)

    if args.dry_run:
        print(f"Generating {args.count} template sentences (--dry-run) ...")
        lines = generate_templates(args.count, seed=args.seed)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {len(lines)} sentences -> {out_path}")
        print("Next: python validate.py")
        return

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    print(f"Generating ~{args.count} passages via OpenAI ({args.model}); "
          f"temperatures cycling {temps} ...")
    records = generate_openai_stories(
        args.count, args.model, args.per_call, args.workers, temps, args.seed
    )

    # 1) raw candidate text (input to validate.py, which salvages valid sentences)
    with open(out_path, "w") as f:
        f.write("\n".join(r["text"] for r in records) + "\n")

    # 2) full per-passage validation log — for manual inspection
    log_path = os.path.join(data_dir, "generation_log.jsonl")
    with open(log_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # 3) flagged file: only passages that used an out-of-vocab token
    flagged = [r for r in records if not r["valid"]]
    flagged_path = os.path.join(data_dir, "flagged_invalid.txt")
    with open(flagged_path, "w") as f:
        for r in flagged:
            f.write(f"# unknown: {', '.join(r['unknown'])}\n{r['text']}\n\n")

    _summarize(records)
    print(f"\nwrote {out_path}  ({len(records)} passages)")
    print(f"wrote {log_path}  (per-passage validation log)")
    print(f"wrote {flagged_path}  ({len(flagged)} passages with out-of-vocab tokens)")
    print("Next: python validate.py  (splits passages into sentences + re-checks)")


if __name__ == "__main__":
    main()
