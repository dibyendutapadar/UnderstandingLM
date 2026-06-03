"""Step 4 — train a tiny "microscope" transformer and export every weight.

Deliberately the smallest transformer that is still a real transformer, so that
the entire forward pass fits on screen as tables of numbers:

    1 layer, 1 head, d_model = 3, MLP hidden = 6, causal self-attention,
    no LayerNorm (keeps the math a clean chain of matmul / add / relu / softmax).

Forward chain (what the frontend re-creates tensor-by-tensor):

    ids -> tok_emb + pos_emb = x
        -> Q = xWq, K = xWk, V = xWv
        -> scores = QKᵀ / sqrt(d)  (causal-masked)
        -> attn = softmax(scores)
        -> attn_out = attn · V
        -> x1 = x + attn_out                      (residual 1)
        -> h  = relu(x1 W1 + b1)
        -> m  = h W2 + b2
        -> x2 = x1 + m                            (residual 2)
        -> logits = x2 Wu + bu  -> softmax = next-token probabilities

Trains next-token prediction on data/sentences.jsonl and writes
data/transformer.json (weights + vocab + config + a reference forward pass that
the JS implementation self-checks against).
"""

import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab as V

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Vocabulary (built from the corpus: core + whatever extras survived filtering)
# ---------------------------------------------------------------------------
def build_vocab(sentences_path):
    freq = Counter()
    with open(sentences_path) as f:
        for line in f:
            freq.update(json.loads(line)["tokens"])
    # core tokens first (stable ids), then extras by frequency
    core = [t for t in V.VOCAB if freq[t] > 0]
    extras = [t for t, _ in freq.most_common() if t not in V.word2idx]
    itos = core + extras
    stoi = {t: i for i, t in enumerate(itos)}
    category = {t: V.token2category.get(t, "other") for t in itos}
    return itos, stoi, category


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, mlp_hidden):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        # vocab_size includes a trailing PAD id (= vocab_size - 1)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, d_model)
        self.unembed = nn.Linear(d_model, vocab_size - 1)  # never predict PAD
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)            # (B,T,d)
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_model)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        x1 = x + attn @ v                                    # residual 1
        x2 = x1 + self.fc2(F.relu(self.fc1(x1)))             # residual 2
        return self.unembed(x2)


# ---------------------------------------------------------------------------
# Training data: pad each sentence to block_size; targets are next tokens.
# ---------------------------------------------------------------------------
def make_batches(sentences_path, stoi, block_size, pad_id):
    X, Y = [], []
    with open(sentences_path) as f:
        for line in f:
            ids = [stoi[t] for t in json.loads(line)["tokens"] if t in stoi]
            ids = ids[:block_size]
            if len(ids) < 2:
                continue
            x = ids + [pad_id] * (block_size - len(ids))
            y = ids[1:] + [pad_id] * (block_size - len(ids) + 1)
            X.append(x)
            Y.append(y[:block_size])
    return torch.tensor(X), torch.tensor(Y)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _mat(linear):
    """Export a Linear as in×out (so JS does y = x @ W + b), plus bias."""
    return {
        "W": linear.weight.detach().t().contiguous().numpy().round(6).tolist(),
        "b": linear.bias.detach().numpy().round(6).tolist(),
    }


def export(model, itos, category, config, sentences_path, stoi, path):
    model.eval()
    w = {
        "tok_emb": model.tok_emb.weight.detach()[: len(itos)].numpy().round(6).tolist(),
        "pos_emb": model.pos_emb.weight.detach().numpy().round(6).tolist(),
        "Wq": _mat(model.Wq), "Wk": _mat(model.Wk), "Wv": _mat(model.Wv),
        "fc1": _mat(model.fc1), "fc2": _mat(model.fc2),
        "unembed": _mat(model.unembed),
    }
    # a reference forward pass (last-position logits) for the JS self-check
    ref_tokens = _first_sentence(sentences_path, stoi)
    with torch.no_grad():
        logits = model(torch.tensor([ref_tokens]))[0, -1]
    payload = {
        "config": config,
        "vocab": itos,
        "categories": category,
        "weights": w,
        "reference": {"tokens": ref_tokens,
                      "logits_last": logits.numpy().round(5).tolist()},
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _first_sentence(sentences_path, stoi):
    with open(sentences_path) as f:
        for line in f:
            ids = [stoi[t] for t in json.loads(line)["tokens"] if t in stoi]
            if len(ids) >= 3:
                return ids[:8]
    raise SystemExit("No usable sentence for the reference forward pass.")


def sample_predictions(model, itos, stoi, block_size, prompts):
    print("\nNext-token predictions:")
    for p in prompts:
        ids = [stoi[t] for t in p.split() if t in stoi]
        if not ids:
            continue
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]
        top = torch.topk(F.softmax(logits, -1), 4)
        preds = ", ".join(f"{itos[i]}({v:.2f})" for v, i in zip(top.values, top.indices))
        print(f"  {p:<22} -> {preds}")


def main():
    ap = argparse.ArgumentParser(description="Train the microscope transformer.")
    ap.add_argument("--sentences", default=os.path.join(HERE, "data", "sentences.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "transformer.json"))
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--mlp-hidden", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if not os.path.exists(args.sentences):
        raise SystemExit(f"No corpus at {args.sentences}. Run validate.py first.")

    itos, stoi, category = build_vocab(args.sentences)
    pad_id = len(itos)                       # PAD sits just past the real vocab
    vocab_size = len(itos) + 1
    print(f"vocab: {len(itos)} tokens (+PAD)  | d={args.dim} head=1 layer=1 "
          f"mlp={args.mlp_hidden} block={args.block_size}")

    X, Y = make_batches(args.sentences, stoi, args.block_size, pad_id)
    print(f"training sequences: {len(X)}")

    model = TinyTransformer(vocab_size, args.dim, args.block_size, args.mlp_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(X)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            logits = model(X[idx])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), Y[idx].reshape(-1),
                ignore_index=pad_id)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {total / n:.4f}")

    config = {"d_model": args.dim, "n_head": 1, "n_layer": 1,
              "mlp_hidden": args.mlp_hidden, "block_size": args.block_size,
              "vocab_size": len(itos)}
    out = export(model, itos, category, config, args.sentences, stoi, args.out)
    print(f"\nwrote {out}  ({len(itos)} tokens)")

    sample_predictions(model, itos, stoi, args.block_size, [
        "the king is", "she is a", "the apple is", "he like the red",
        "king and queen are", "the boy",
    ])


if __name__ == "__main__":
    main()
