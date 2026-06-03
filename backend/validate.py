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
import re
from collections import Counter

import vocab as V


TERMINATORS = {".", "?"}

# Detach punctuation that models tend to glue onto words ("king." -> "king" ".").
# Only ".,?" are core vocab; others (e.g. "!") become sparse extras instead of
# silently corrupting the word they were stuck to.
_PUNCT_RE = re.compile(r'([.,!?;:"()])')


def normalize(text):
    """Lowercase and space-separate punctuation so tokens line up with the vocab."""
    return " ".join(_PUNCT_RE.sub(r" \1 ", text.lower()).split())


def tokenize(line):
    return normalize(line).split()


def split_sentences(tokens):
    """Split a token stream into sentences at '.' / '?' (terminator kept).

    Lets us feed multi-sentence story lines through the same checker as single
    template sentences, and salvage the valid sentences out of a story that
    contains one stray out-of-vocab word."""
    sents, cur = [], []
    for t in tokens:
        cur.append(t)
        if t in TERMINATORS:
            sents.append(cur)
            cur = []
    if cur:  # trailing fragment with no terminator
        sents.append(cur)
    return sents


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
    ap.add_argument("--strict", action="store_true",
                    help="reject any sentence with an out-of-vocab token "
                         "(default: keep them; the extra words become a sparse tail)")
    ap.add_argument("--top-tokens", type=int, default=0,
                    help="keep only sentences made entirely of the N most frequent "
                         "tokens (0 = no cap). Caps the vocabulary and drops "
                         "rare-word noise before embedding.")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"No candidates at {args.inp}. Run generate_data.py first.")

    # ---- Pass 1: accept candidate sentences + count all-token frequencies ----
    accepted = []          # list of token lists
    all_freq = Counter()   # over core + extra, used to pick the top-N tokens
    n_rejected = 0
    rejection_examples = []
    for line in open(args.inp):
        if not line.strip():
            continue
        for tokens in split_sentences(tokenize(line)):
            if not tokens:
                continue
            bad = [t for t in tokens if t not in V.word2idx]
            #   strict  -> any out-of-vocab token rejects the sentence
            #   relaxed -> keep it, as long as it has >=1 core token
            core_present = len(bad) < len(tokens)
            if (args.strict and bad) or (not args.strict and not core_present):
                n_rejected += 1
                if len(rejection_examples) < 10:
                    rejection_examples.append(
                        {"text": " ".join(tokens), "unknown": sorted(set(bad))})
                continue
            accepted.append(tokens)
            all_freq.update(tokens)

    # ---- Optional vocabulary cap: keep only sentences built from the top N ----
    allowed = None
    n_dropped_by_cap = 0
    if args.top_tokens > 0:
        allowed = {w for w, _ in all_freq.most_common(args.top_tokens)}
        kept = [s for s in accepted if all(t in allowed for t in s)]
        n_dropped_by_cap = len(accepted) - len(kept)
        accepted = kept

    # ---- Pass 2: write kept sentences + gather stats over the final corpus ----
    token_freq = Counter()      # core tokens
    extra_freq = Counter()      # kept words outside the core
    n_with_extra = 0
    rng = random.Random(0)
    SAMPLE_K = 24
    samples = []
    with open(args.out, "w") as fout:
        for tokens in accepted:
            token_freq.update(t for t in tokens if t in V.word2idx)
            extras = [t for t in tokens if t not in V.word2idx]
            if extras:
                n_with_extra += 1
                extra_freq.update(extras)
            text = " ".join(tokens)
            fout.write(json.dumps({"text": text, "tokens": tokens}) + "\n")
            if len(samples) < SAMPLE_K:
                samples.append(text)
            elif rng.random() < SAMPLE_K / len(samples):
                samples[rng.randrange(SAMPLE_K)] = text

    n_valid = len(accepted)
    cat_freq = Counter()
    for tok, c in token_freq.items():
        cat_freq[V.token2category[tok]] += c
    missing = [t for t in V.VOCAB if token_freq[t] == 0]
    total = n_valid + n_rejected

    stats = {
        "strict": args.strict,
        "top_tokens": args.top_tokens,
        "num_sentences": n_valid,
        "num_tokens_emitted": sum(token_freq.values()),
        "vocab_size": len(V.VOCAB),
        "vocab_covered": len(V.VOCAB) - len(missing),
        "missing_tokens": missing,
        "rejected": n_rejected,
        "dropped_by_cap": n_dropped_by_cap,
        "rejection_rate": round(n_rejected / total, 4) if total else 0,
        "rejection_examples": rejection_examples,
        "token_freq": {t: token_freq[t] for t in V.VOCAB},
        "category_freq": {c: cat_freq[c] for c in V.CATEGORIES},
        "samples": samples,
        "sentences_with_extra": n_with_extra,
        "num_extra_types": len(extra_freq),
        "num_extra_tokens": sum(extra_freq.values()),
        "extra_freq": dict(extra_freq.most_common(150)),
    }
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    # ---- console summary ----
    print(f"mode            : {'strict' if args.strict else 'relaxed (extras kept)'}")
    if args.top_tokens:
        kept_vocab = len(token_freq) + len(extra_freq)
        print(f"top-tokens cap  : {args.top_tokens} -> kept vocab {kept_vocab}, "
              f"dropped {n_dropped_by_cap} sentences over the cap")
        # warn if any analogy word didn't make the cut
        cut = [w for a in V.ANALOGIES for w in (a['a'], a['b'], a['c'], a['expect'])
               if allowed and w not in allowed]
        if cut:
            print(f"  WARNING: analogy words outside top-{args.top_tokens}: {sorted(set(cut))}")
    print(f"valid sentences : {n_valid}")
    print(f"rejected        : {n_rejected} ({stats['rejection_rate'] * 100:.2f}%)")
    print(f"core coverage   : {stats['vocab_covered']}/{len(V.VOCAB)}")
    if missing:
        print(f"  MISSING core tokens (never appear): {missing}")
    if extra_freq:
        print(f"extra vocabulary: {len(extra_freq)} word types kept, "
              f"{sum(extra_freq.values())} tokens, in {n_with_extra} sentences")
        top = ", ".join(f"{w}({c})" for w, c in extra_freq.most_common(12))
        print(f"  most common extras: {top}")
    print(f"wrote {args.out}")
    print(f"wrote {args.stats}")

    if n_valid == 0:
        raise SystemExit("No valid sentences — check the generator / cap.")


if __name__ == "__main__":
    main()
