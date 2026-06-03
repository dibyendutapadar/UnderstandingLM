// In-browser forward pass for the microscope transformer. Mirrors
// backend/train_transformer.py exactly and returns EVERY intermediate tensor so
// the UI can show the whole computation as tables of numbers.
//
// Shapes: T = sequence length, d = d_model, h = mlp hidden, Vr = real vocab.

import model from "../data/transformer.json";

export const config = model.config;
export const vocab = model.vocab; // index -> token string
export const categories = model.categories;
export const stoi = Object.fromEntries(vocab.map((t, i) => [t, i]));

const W = model.weights;
const D = config.d_model;

// ---- tiny matrix helpers (everything here is small) ----
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

// y = x · W + b   where W is in×out, x is length-in, returns length-out
function linear(x, { W: w, b }) {
  const out = b.slice();
  for (let o = 0; o < out.length; o++) {
    let s = b[o];
    for (let i = 0; i < x.length; i++) s += x[i] * w[i][o];
    out[o] = s;
  }
  return out;
}

const addVec = (a, b) => a.map((v, i) => v + b[i]);
const relu = (a) => a.map((v) => (v > 0 ? v : 0));

function softmax(a) {
  const m = Math.max(...a);
  const ex = a.map((v) => Math.exp(v - m));
  const s = ex.reduce((p, q) => p + q, 0);
  return ex.map((v) => v / s);
}

/**
 * Run the full forward pass on an array of token ids.
 * Returns all intermediate tensors (arrays of rows, one row per position).
 */
export function forward(ids) {
  const T = ids.length;
  const pos = ids.map((_, t) => t);

  const tokEmb = ids.map((id) => W.tok_emb[id].slice());
  const posEmb = pos.map((t) => W.pos_emb[t].slice());
  const x = tokEmb.map((e, t) => addVec(e, posEmb[t])); // input to the block

  const Q = x.map((row) => linear(row, W.Wq));
  const K = x.map((row) => linear(row, W.Wk));
  const Vv = x.map((row) => linear(row, W.Wv));

  // scores = Q·Kᵀ / sqrt(d), causal-masked, then softmax row-wise
  const scale = Math.sqrt(D);
  const scores = []; // raw scaled scores (with -Inf above the diagonal)
  const attn = []; // softmax weights
  for (let i = 0; i < T; i++) {
    const rowRaw = [];
    for (let j = 0; j < T; j++) {
      rowRaw.push(j <= i ? dot(Q[i], K[j]) / scale : -Infinity);
    }
    scores.push(rowRaw);
    attn.push(softmax(rowRaw));
  }

  // attn_out[i] = Σ_j attn[i][j] · V[j]
  const attnOut = attn.map((wRow) => {
    const acc = new Array(D).fill(0);
    for (let j = 0; j < T; j++)
      for (let c = 0; c < D; c++) acc[c] += wRow[j] * Vv[j][c];
    return acc;
  });

  const resid1 = x.map((row, t) => addVec(row, attnOut[t])); // residual 1
  const mlpHidden = resid1.map((row) => linear(row, W.fc1));
  const mlpRelu = mlpHidden.map(relu);
  const mlpOut = mlpRelu.map((row) => linear(row, W.fc2));
  const resid2 = resid1.map((row, t) => addVec(row, mlpOut[t])); // residual 2

  const logits = resid2.map((row) => linear(row, W.unembed));
  const probs = logits.map(softmax);

  return {
    tokens: ids.map((id) => vocab[id]),
    ids,
    tokEmb, posEmb, x,
    Q, K, V: Vv,
    scores, attn, attnOut,
    resid1, mlpHidden, mlpRelu, mlpOut, resid2,
    logits, probs,
  };
}

// Top-k next-token predictions from the last position.
export function nextToken(out, k = 8) {
  const last = out.probs[out.probs.length - 1];
  return last
    .map((p, i) => ({ token: vocab[i], category: categories[vocab[i]], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k);
}

// Self-check against the Python reference forward pass (max abs logit diff).
export function parityError() {
  const ref = model.reference;
  const out = forward(ref.tokens);
  const got = out.logits[out.logits.length - 1];
  let maxErr = 0;
  for (let i = 0; i < got.length; i++)
    maxErr = Math.max(maxErr, Math.abs(got[i] - ref.logits_last[i]));
  return maxErr;
}
