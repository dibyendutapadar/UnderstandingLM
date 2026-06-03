"""Dump every passage from generation_log.jsonl into raw_sentences.txt.

Use this to rebuild the raw corpus from a previous OpenAI generation WITHOUT
re-calling the API, and without any validity filtering — every passage is kept
(punctuation is normalized so the tokens line up with the vocab; the relaxed
validate.py downstream keeps out-of-vocab words as a sparse tail).
"""

import argparse
import json
import os

from validate import normalize

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="generation_log.jsonl -> raw_sentences.txt")
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(HERE, "data", "generation_log.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "raw_sentences.txt"))
    ap.add_argument("--raw", action="store_true",
                    help="dump the text verbatim (skip punctuation normalization)")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"No log at {args.inp}. Run generate_data.py (OpenAI) first.")

    n = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            if not args.raw:
                text = normalize(text)
            if text:
                fout.write(text + "\n")
                n += 1

    print(f"dumped {n} passages -> {args.out}"
          f"{'' if args.raw else ' (punctuation normalized)'}")
    print("Next: python validate.py   (relaxed — keeps out-of-vocab words as extras)")


if __name__ == "__main__":
    main()
