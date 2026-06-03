"""Step 3 — train a 3D skip-gram embedding and verify the analogies.

Architecture (deliberately tiny, exactly the shape we can visualize):

    word id --> Embedding(50 -> 3) --> Linear(3 -> 50) --> softmax(context)

We train it to predict neighboring words (skip-gram). After training, each of
the 50 tokens has a learned (x, y, z) coordinate, which the frontend plots
directly (no PCA/t-SNE — the space really is 3D).

Outputs:
    data/embeddings.json            [{word, x, y, z, category}]
    data/embeddings_snapshots.json  (only with --snapshots) per-epoch frames
                                    for a future "watch embeddings form" slider

The script also prints an analogy report (king - man + woman = ?, etc.) and a
category-cohesion report. Those are our objective pass/fail for the step.
"""

import argparse
import json
import math
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import vocab as V

HERE = os.path.dirname(os.path.abspath(__file__))


def _keep_probs(counts, total, t):
    """word2vec frequent-word subsampling: rarer words kept ~always, very
    frequent function words ('the', 'is', '.') kept with low probability so
    they stop dominating the gradients."""
    probs = {}
    for tok, c in counts.items():
        f = c / total
        probs[tok] = 1.0 if f <= t else math.sqrt(t / f)
    return probs


# ---------------------------------------------------------------------------
# Data: build (center, context) skip-gram pairs from sentences.jsonl
# ---------------------------------------------------------------------------
def build_pairs(sentences_path, window, subsample_t, seed):
    rng = random.Random(seed)
    sentences = []
    counts = Counter()
    with open(sentences_path) as f:
        for line in f:
            toks = json.loads(line)["tokens"]
            sentences.append(toks)
            counts.update(toks)
    total = sum(counts.values())
    keep = _keep_probs(counts, total, subsample_t) if subsample_t > 0 else None

    centers, contexts = [], []
    for toks in sentences:
        # subsample frequent words out of this sentence first
        if keep is not None:
            toks = [w for w in toks if rng.random() < keep[w]]
        ids = [V.word2idx[t] for t in toks]
        n = len(ids)
        for i in range(n):
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                centers.append(ids[i])
                contexts.append(ids[j])
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        return self.fc(self.embedding(x))


# ---------------------------------------------------------------------------
# Analysis helpers (numpy, on the learned embedding matrix)
# ---------------------------------------------------------------------------
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def nearest(emb, vec, exclude=(), k=3):
    """Top-k tokens by cosine similarity to vec."""
    sims = []
    for w, i in V.word2idx.items():
        if w in exclude:
            continue
        sims.append((w, float(np.dot(_unit(emb[i]), _unit(vec)))))
    sims.sort(key=lambda t: -t[1])
    return sims[:k]


def analogy_report(emb):
    print("\nAnalogy checks  (a - b + c  ~=  expected):")
    hits = 0
    for a in V.ANALOGIES:
        vec = emb[V.word2idx[a["a"]]] - emb[V.word2idx[a["b"]]] + emb[V.word2idx[a["c"]]]
        top = nearest(emb, vec, exclude=(a["a"], a["b"], a["c"]), k=3)
        got = top[0][0]
        ok = got == a["expect"]
        hits += ok
        ranked = ", ".join(f"{w}({s:.2f})" for w, s in top)
        mark = "OK " if ok else "·  "
        print(f"  {mark}{a['a']} - {a['b']} + {a['c']} = {a['expect']:<6} -> {ranked}")
    print(f"  => {hits}/{len(V.ANALOGIES)} analogies top-1 correct")
    return hits


def cohesion_report(emb):
    """Mean intra-category vs inter-category cosine — is structure forming?"""
    units = {w: _unit(emb[i]) for w, i in V.word2idx.items()}
    intra, inter = [], []
    words = list(V.word2idx)
    for i, a in enumerate(words):
        for b in words[i + 1:]:
            s = float(np.dot(units[a], units[b]))
            (intra if V.token2category[a] == V.token2category[b] else inter).append(s)
    print("\nCategory cohesion (cosine):")
    print(f"  mean intra-category: {np.mean(intra):+.3f}")
    print(f"  mean inter-category: {np.mean(inter):+.3f}")
    print(f"  separation (intra - inter): {np.mean(intra) - np.mean(inter):+.3f}")


def export_embeddings(emb, path):
    rows = []
    for w, i in V.word2idx.items():
        x, y, z = emb[i]
        rows.append({
            "word": w,
            "x": round(float(x), 5),
            "y": round(float(y), 5),
            "z": round(float(z), 5),
            "category": V.token2category[w],
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the 3D skip-gram embedding.")
    ap.add_argument("--sentences", default=os.path.join(HERE, "data", "sentences.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "embeddings.json"))
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--subsample", type=float, default=1e-3,
                    help="frequent-word subsampling threshold (0 disables)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--snapshots", action="store_true",
                    help="also save per-epoch frames for the training slider")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.sentences):
        raise SystemExit(f"No corpus at {args.sentences}. Run generate_data.py + validate.py.")

    print(f"Building skip-gram pairs (window={args.window}, subsample={args.subsample}) ...")
    centers, contexts = build_pairs(args.sentences, args.window, args.subsample, args.seed)
    print(f"  {len(centers):,} training pairs")

    model = SkipGram(len(V.VOCAB), args.dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(centers)
    snap_epochs = sorted(set([0, 1, 2, 5, 10, 20, 40, 80, args.epochs - 1]))
    snapshots = []

    def current_emb():
        return model.embedding.weight.detach().cpu().numpy().copy()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            x, y = centers[idx], contexts[idx]
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        avg = total / n
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {avg:.4f}")
        if args.snapshots and epoch in snap_epochs:
            snapshots.append({"epoch": epoch, "loss": round(avg, 4),
                              "emb": current_emb().round(5).tolist()})

    emb = current_emb()
    rows = export_embeddings(emb, args.out)
    print(f"\nwrote {args.out}  ({len(rows)} tokens, dim={args.dim})")

    if args.snapshots:
        snap_path = os.path.join(HERE, "data", "embeddings_snapshots.json")
        with open(snap_path, "w") as f:
            json.dump({"vocab": V.VOCAB,
                       "categories": V.token2category,
                       "frames": snapshots}, f)
        print(f"wrote {snap_path}  ({len(snapshots)} frames)")

    analogy_report(emb)
    cohesion_report(emb)


if __name__ == "__main__":
    main()
