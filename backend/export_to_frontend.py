"""Copy trained/generated artifacts into frontend/src/data/.

These JSON files are the contract the static frontend imports. Re-run whenever
the vocabulary, corpus stats, or embeddings change. Missing artifacts (e.g.
embeddings.json before Step 3 has run) are skipped with a note.
"""

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEST = os.path.join(ROOT, "frontend", "src", "data")

ARTIFACTS = [
    os.path.join(ROOT, "shared", "vocab.json"),
    os.path.join(HERE, "data", "corpus_stats.json"),
    os.path.join(HERE, "data", "embeddings.json"),
]


def main():
    os.makedirs(DEST, exist_ok=True)
    for src in ARTIFACTS:
        name = os.path.basename(src)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DEST, name))
            print(f"copied {name} -> frontend/src/data/")
        else:
            print(f"skip   {name} (not generated yet)")


if __name__ == "__main__":
    main()
