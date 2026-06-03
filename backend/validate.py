"""Step 2 — the checker.

Reads candidate sentences (data/raw_sentences.txt), keeps only those whose
every token is in the 50-token vocabulary, and writes:

  data/sentences.jsonl   one {"text","tokens"} record per valid sentence
  data/corpus_stats.json per-token + per-category frequencies, coverage,
                         rejection summary (consumed by the frontend Data page)

Run after generate_data.py.
"""

import argparse
import json
import os
import random
from collections import Counter

import vocab as V


def tokenize(line):
    return line.strip().split()


def validate_sentence(tokens):
    """Return (ok, bad_tokens). A sentence is valid iff every token is known."""
    bad = [t for t in tokens if t not in V.word2idx]
    return (len(bad) == 0 and len(tokens) > 0), bad


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Validate + summarize the corpus.")
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(here, "data", "raw_sentences.txt"))
    ap.add_argument("--out", default=os.path.join(here, "data", "sentences.jsonl"))
    ap.add_argument("--stats", default=os.path.join(here, "data", "corpus_stats.json"))
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"No candidates at {args.inp}. Run generate_data.py first.")

    token_freq = Counter()
    n_valid = 0
    n_rejected = 0
    rejection_examples = []
    rng = random.Random(0)
    SAMPLE_K = 24
    samples = []  # reservoir sample of valid sentences (for the frontend)

    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            tokens = tokenize(line)
            ok, bad = validate_sentence(tokens)
            if not ok:
                n_rejected += 1
                if len(rejection_examples) < 10:
                    rejection_examples.append(
                        {"text": line.strip(), "unknown": sorted(set(bad))}
                    )
                continue
            n_valid += 1
            token_freq.update(tokens)
            text = " ".join(tokens)
            fout.write(json.dumps({"text": text, "tokens": tokens}) + "\n")
            # reservoir sample for the frontend Data page
            if len(samples) < SAMPLE_K:
                samples.append(text)
            elif rng.random() < SAMPLE_K / n_valid:
                samples[rng.randrange(SAMPLE_K)] = text

    # per-category aggregation
    cat_freq = Counter()
    for tok, c in token_freq.items():
        cat_freq[V.token2category[tok]] += c

    missing = [t for t in V.VOCAB if token_freq[t] == 0]
    total_candidates = n_valid + n_rejected

    stats = {
        "num_sentences": n_valid,
        "num_tokens_emitted": sum(token_freq.values()),
        "vocab_size": len(V.VOCAB),
        "vocab_covered": len(V.VOCAB) - len(missing),
        "missing_tokens": missing,
        "rejected": n_rejected,
        "rejection_rate": round(n_rejected / total_candidates, 4) if total_candidates else 0,
        "rejection_examples": rejection_examples,
        "token_freq": {t: token_freq[t] for t in V.VOCAB},  # vocab order, incl zeros
        "category_freq": {c: cat_freq[c] for c in V.CATEGORIES},
        "samples": samples,
    }
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    # ---- console summary ----
    print(f"valid sentences : {n_valid}")
    print(f"rejected        : {n_rejected} ({stats['rejection_rate'] * 100:.2f}%)")
    print(f"vocab coverage  : {stats['vocab_covered']}/{len(V.VOCAB)}")
    if missing:
        print(f"  MISSING tokens (never appear): {missing}")
    if rejection_examples:
        print("  sample rejections:")
        for ex in rejection_examples[:3]:
            print(f"    {ex['text']!r}  unknown={ex['unknown']}")
    print(f"wrote {args.out}")
    print(f"wrote {args.stats}")

    if n_valid == 0:
        raise SystemExit("No valid sentences — check the generator.")


if __name__ == "__main__":
    main()
