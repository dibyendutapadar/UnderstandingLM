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

# Effective vocabulary = core 50 (fixed ids 0..49) + a sparse tail of extra
# words discovered in the corpus (ids 50+, category "other"). Rebuilt from the
# corpus by build_vocabulary(); the analysis/training functions read these.
WORDS = list(V.VOCAB)
WORD2IDX = dict(V.word2idx)
CATEGORY = dict(V.token2category)


def build_vocabulary(sentences_path, min_extra_count=1):
    """Keep the core 50 plus any out-of-core word appearing >= min_extra_count
    times. Updates the module-level effective vocabulary; returns the extras."""
    extra = Counter()
    with open(sentences_path) as f:
        for line in f:
            for t in json.loads(line)["tokens"]:
                if t not in V.word2idx:
                    extra[t] += 1
    extras = [w for w, c in extra.most_common() if c >= min_extra_count]
    global WORDS, WORD2IDX, CATEGORY
    WORDS = list(V.VOCAB) + extras
    WORD2IDX = {w: i for i, w in enumerate(WORDS)}
    CATEGORY = {**V.token2category, **{w: "other" for w in extras}}
    return extras, extra


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
        ids = [WORD2IDX[t] for t in toks if t in WORD2IDX]
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
    """Top-k tokens by cosine similarity to vec (over the effective vocab)."""
    sims = []
    for w, i in WORD2IDX.items():
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


def _separation(emb):
    """Mean intra- minus inter-category cosine, over the core 50 only (the
    sparse 'other' extras aren't a semantic category)."""
    core = list(V.VOCAB)
    units = {w: _unit(emb[WORD2IDX[w]]) for w in core}
    intra, inter = [], []
    for i, a in enumerate(core):
        for b in core[i + 1:]:
            s = float(np.dot(units[a], units[b]))
            (intra if V.token2category[a] == V.token2category[b] else inter).append(s)
    return np.mean(intra), np.mean(inter)


def cohesion_report(emb):
    """Mean intra-category vs inter-category cosine — is structure forming?"""
    intra, inter = _separation(emb)
    print("\nCategory cohesion (cosine):")
    print(f"  mean intra-category: {intra:+.3f}")
    print(f"  mean inter-category: {inter:+.3f}")
    print(f"  separation (intra - inter): {intra - inter:+.3f}")


def export_embeddings(emb, path):
    rows = []
    for w, i in WORD2IDX.items():
        x, y, z = emb[i]
        rows.append({
            "word": w,
            "x": round(float(x), 5),
            "y": round(float(y), 5),
            "z": round(float(z), 5),
            "category": CATEGORY[w],
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


# ---------------------------------------------------------------------------
# Method 1 — skip-gram (SGD). Returns (embedding matrix, snapshots).
# ---------------------------------------------------------------------------
def train_skipgram(sentences_path, dim, window, subsample, epochs, batch, lr,
                   seed, snapshots=False, verbose=True):
    torch.manual_seed(seed)
    centers, contexts = build_pairs(sentences_path, window, subsample, seed)
    if verbose:
        print(f"  {len(centers):,} skip-gram pairs (window={window}, subsample={subsample})")
    model = SkipGram(len(WORDS), dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(centers)
    snap_epochs = sorted(set([0, 1, 2, 5, 10, 20, 40, 80, epochs - 1]))
    frames = []

    def emb():
        return model.embedding.weight.detach().cpu().numpy().copy()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            logits = model(centers[idx])
            loss = criterion(logits, contexts[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss {total / n:.4f}")
        if snapshots and epoch in snap_epochs:
            frames.append({"epoch": epoch, "loss": round(total / n, 4),
                           "emb": emb().round(5).tolist()})
    return emb(), frames


# ---------------------------------------------------------------------------
# Method 2 — count-based PPMI + truncated SVD (LSA / Hellinger-PCA style).
# The top-`dim` singular vectors are the best linear `dim`-D summary of the
# word/context association structure. Deterministic; strong on tiny corpora.
# ---------------------------------------------------------------------------
def build_cooccurrence(sentences_path, window):
    size = len(WORDS)
    C = np.zeros((size, size), dtype=np.float64)
    with open(sentences_path) as f:
        for line in f:
            ids = [WORD2IDX[t] for t in json.loads(line)["tokens"] if t in WORD2IDX]
            n = len(ids)
            for i in range(n):
                lo, hi = max(0, i - window), min(n, i + window + 1)
                for j in range(lo, hi):
                    if j != i:
                        C[ids[i], ids[j]] += 1.0 / abs(i - j)  # distance-weighted
    return C


def ppmi_svd(C, dim, alpha=0.75):
    total = C.sum()
    Pwc = C / total
    Pw = C.sum(1) / total
    ctx = C.sum(0) ** alpha          # context-distribution smoothing (SGNS trick)
    Pc = ctx / ctx.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(Pwc / np.outer(Pw, Pc))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    U, S, _ = np.linalg.svd(ppmi)
    return U[:, :dim] * np.sqrt(S[:dim])   # scale by sqrt(singular values)


# ---------------------------------------------------------------------------
# Method 3 — train high-dim skip-gram, then PCA down to `dim`.
# ---------------------------------------------------------------------------
def pca_to(emb, dim):
    X = emb - emb.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:dim].T


def train_highd_pca(sentences_path, dim, highd, window, subsample, epochs,
                    batch, lr, seed):
    emb, _ = train_skipgram(sentences_path, highd, window, subsample, epochs,
                            batch, lr, seed, verbose=False)
    print(f"  trained {highd}-d skip-gram, projecting to {dim}-d via PCA")
    return pca_to(emb, dim)


METHODS = ("skipgram", "ppmi-svd", "highd-pca")


def build_method(method, args):
    if method == "skipgram":
        emb, frames = train_skipgram(
            args.sentences, args.dim, args.window, args.subsample,
            args.epochs, args.batch, args.lr, args.seed,
            snapshots=args.snapshots)
        return emb, frames
    if method == "ppmi-svd":
        C = build_cooccurrence(args.sentences, args.window)
        return ppmi_svd(C, args.dim, args.alpha), []
    if method == "highd-pca":
        return train_highd_pca(args.sentences, args.dim, args.highd, args.window,
                               args.subsample, args.epochs, args.batch, args.lr,
                               args.seed), []
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a 3D word embedding.")
    ap.add_argument("--method", choices=METHODS, default="ppmi-svd")
    ap.add_argument("--compare", action="store_true",
                    help="run all methods and print an analogy/cohesion comparison")
    ap.add_argument("--sentences", default=os.path.join(HERE, "data", "sentences.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "embeddings.json"))
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--subsample", type=float, default=1e-3,
                    help="skip-gram frequent-word subsampling threshold (0 disables)")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="ppmi-svd context-distribution smoothing exponent")
    ap.add_argument("--highd", type=int, default=48,
                    help="hidden dim for the highd-pca method")
    ap.add_argument("--min-extra-count", type=int, default=1,
                    help="keep out-of-core words appearing >= this many times "
                         "(the sparse tail); raise it to drop rare noise")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--snapshots", action="store_true",
                    help="(skipgram only) save per-epoch frames for the training slider")
    args = ap.parse_args()

    np.random.seed(args.seed)
    if not os.path.exists(args.sentences):
        raise SystemExit(f"No corpus at {args.sentences}. Run generate_data.py + validate.py.")

    extras, extra_counts = build_vocabulary(args.sentences, args.min_extra_count)
    print(f"Vocabulary: {len(V.VOCAB)} core + {len(extras)} sparse extras "
          f"= {len(WORDS)} tokens (min count {args.min_extra_count})")
    if extras:
        preview = ", ".join(f"{w}({extra_counts[w]})" for w in extras[:10])
        print(f"  extras (most frequent): {preview}")

    if args.compare:
        print(f"Comparing methods on {args.sentences} (window={args.window}) ...\n")
        results = {}
        for m in METHODS:
            print(f"[{m}]")
            emb, _ = build_method(m, args)
            results[m] = emb
            analogy_report(emb)
            cohesion_report(emb)
            print()
        print("=" * 56)
        print(f"{'method':<12}{'analogies':>12}{'separation':>14}")
        for m, emb in results.items():
            hits = sum(
                nearest(emb,
                        emb[WORD2IDX[a['a']]] - emb[WORD2IDX[a['b']]] + emb[WORD2IDX[a['c']]],
                        exclude=(a['a'], a['b'], a['c']), k=1)[0][0] == a['expect']
                for a in V.ANALOGIES)
            intra, inter = _separation(emb)
            print(f"{m:<12}{hits:>8}/{len(V.ANALOGIES):<3}{intra - inter:>+14.3f}")
        print("\nPick one with --method NAME (no --compare) to write embeddings.json.")
        return

    print(f"Method: {args.method}")
    emb, frames = build_method(args.method, args)
    rows = export_embeddings(emb, args.out)
    print(f"\nwrote {args.out}  ({len(rows)} tokens, dim={args.dim}, method={args.method})")

    if frames:
        snap_path = os.path.join(HERE, "data", "embeddings_snapshots.json")
        with open(snap_path, "w") as f:
            json.dump({"vocab": V.VOCAB, "categories": V.token2category,
                       "frames": frames}, f)
        print(f"wrote {snap_path}  ({len(frames)} frames)")

    analogy_report(emb)
    cohesion_report(emb)


if __name__ == "__main__":
    main()
