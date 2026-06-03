import { useMemo, useState } from "react";
import {
  forward, nextToken, parityError, vocab, categories, stoi, config,
} from "../lib/transformer.js";
import { categoryColor } from "../lib/categories.js";
import MatrixTable from "../components/MatrixTable.jsx";
import AttentionHeatmap from "../components/AttentionHeatmap.jsx";

const PARITY = parityError();
const D = config.d_model;
const dLabels = Array.from({ length: D }, (_, i) => `d${i}`);
const hLabels = Array.from({ length: config.mlp_hidden }, (_, i) => `n${i}`);

// group vocab by category for the "add token" picker
const GROUPS = {};
vocab.forEach((t) => ((GROUPS[categories[t]] ||= []).push(t)));

const PRESETS = [
  "he like the red apple",
  "the king is",
  "king and queen are",
  "she is a",
  "the boy is very happy",
];

function tokenize(text) {
  return text.split(/\s+/).filter((t) => t in stoi).slice(0, config.block_size);
}

export default function MicroscopePage() {
  const [tokens, setTokens] = useState(() => tokenize("he like the red apple"));
  const ids = tokens.map((t) => stoi[t]);
  const out = useMemo(() => (ids.length ? forward(ids) : null), [tokens.join(" ")]);
  const preds = out ? nextToken(out, 8) : [];

  const add = (t) => setTokens((ts) => ts.length < config.block_size ? [...ts, t] : ts);
  const removeAt = (i) => setTokens((ts) => ts.filter((_, k) => k !== i));

  return (
    <div>
      <h2>Transformer Microscope</h2>
      <p className="muted" style={{ maxWidth: 820 }}>
        A real transformer — 1 layer, 1 head, {D}-dim, MLP hidden {config.mlp_hidden},
        {" "}{config.vocab_size} tokens — small enough to read every number. Build a
        sentence and watch the entire forward pass: embeddings → Q/K/V → attention →
        MLP → next-token probabilities. The math runs in your browser and matches
        PyTorch to <code>{PARITY.toExponential(1)}</code>.
      </p>

      {/* ---- sentence builder ---- */}
      <div className="card">
        <div className="builder">
          {tokens.map((t, i) => (
            <span key={i} className="tok-chip" style={{ borderColor: categoryColor(categories[t]) }}>
              <span className="muted pos">{i}</span>
              {t}
              <button onClick={() => removeAt(i)} title="remove">×</button>
            </span>
          ))}
          <select className="wsel add-sel" value="" onChange={(e) => e.target.value && add(e.target.value)}>
            <option value="">+ add token…</option>
            {Object.entries(GROUPS).map(([cat, toks]) => (
              <optgroup key={cat} label={cat}>
                {toks.map((t) => <option key={t} value={t}>{t}</option>)}
              </optgroup>
            ))}
          </select>
          {tokens.length > 0 && (
            <button className="ghost" onClick={() => setTokens([])}>clear</button>
          )}
        </div>
        <div className="presets" style={{ marginTop: 10 }}>
          {PRESETS.map((p) => (
            <button key={p} className="preset" onClick={() => setTokens(tokenize(p))}>{p}</button>
          ))}
        </div>
      </div>

      {!out ? (
        <div className="notice" style={{ marginTop: 20 }}>Add a token to run the model.</div>
      ) : (
        <>
          {/* ---- prediction ---- */}
          <h3 style={{ marginTop: 28 }}>Predicted next token</h3>
          <div className="card">
            <div className="muted" style={{ marginBottom: 10, fontFamily: "ui-monospace" }}>
              {tokens.join(" ")} <span style={{ color: "var(--accent)" }}>→ {preds[0].token}</span>
            </div>
            {preds.map((p) => (
              <div key={p.token} className="nbr">
                <span className="dot" style={{ background: categoryColor(p.category) }} />
                <code>{p.token}</code>
                <span className="freq-bar" style={{ margin: "0 8px" }}>
                  <span className="freq-fill" style={{ width: `${p.p * 100}%`, background: categoryColor(p.category) }} />
                </span>
                <span className="muted mono">{(p.p * 100).toFixed(1)}%</span>
              </div>
            ))}
            <button className="primary" style={{ marginTop: 12 }}
              onClick={() => add(preds[0].token)}
              disabled={tokens.length >= config.block_size}>
              Append “{preds[0].token}” and re-run →
            </button>
          </div>

          {/* ---- the forward pass ---- */}
          <h3 style={{ marginTop: 28 }}>1 · Embeddings</h3>
          <p className="muted pipe-note">Each token id is looked up, plus a position vector. Their sum enters the block.</p>
          <div className="matrix-row">
            <MatrixTable rows={out.tokEmb} rowLabels={out.tokens} colLabels={dLabels} title="Token embedding" />
            <MatrixTable rows={out.posEmb} rowLabels={out.tokens} colLabels={dLabels} title="Position embedding" />
            <MatrixTable rows={out.x} rowLabels={out.tokens} colLabels={dLabels} title="Input  x = tok + pos" />
          </div>

          <h3 style={{ marginTop: 28 }}>2 · Attention</h3>
          <p className="muted pipe-note">Each token produces a Query, Key and Value. Q·Kᵀ (scaled, causal-masked) → softmax = who attends to whom.</p>
          <div className="matrix-row">
            <MatrixTable rows={out.Q} rowLabels={out.tokens} colLabels={dLabels} title="Q (query)" />
            <MatrixTable rows={out.K} rowLabels={out.tokens} colLabels={dLabels} title="K (key)" />
            <MatrixTable rows={out.V} rowLabels={out.tokens} colLabels={dLabels} title="V (value)" />
          </div>
          <div className="matrix-row">
            <AttentionHeatmap attn={out.attn} tokens={out.tokens} />
            <MatrixTable rows={out.attnOut} rowLabels={out.tokens} colLabels={dLabels} title="Attention output = attn · V" />
            <MatrixTable rows={out.resid1} rowLabels={out.tokens} colLabels={dLabels} title="Residual 1 = x + attn out" hint="add back input" />
          </div>

          <h3 style={{ marginTop: 28 }}>3 · MLP</h3>
          <p className="muted pipe-note">A 2-layer MLP with ReLU processes each position independently. Watch for dead (zero) neurons after ReLU.</p>
          <div className="matrix-row">
            <MatrixTable rows={out.mlpHidden} rowLabels={out.tokens} colLabels={hLabels} title="Hidden = x₁W₁+b₁" />
            <MatrixTable rows={out.mlpRelu} rowLabels={out.tokens} colLabels={hLabels} title="ReLU(hidden)" />
            <MatrixTable rows={out.resid2} rowLabels={out.tokens} colLabels={dLabels} title="Residual 2 = x₁ + MLP out" />
          </div>

          <h3 style={{ marginTop: 28 }}>4 · Output</h3>
          <p className="muted pipe-note">
            The last row of Residual 2 is projected to {config.vocab_size} logits and softmaxed — that's the
            prediction shown above.
          </p>
        </>
      )}
    </div>
  );
}
