import { useMemo, useState } from "react";
import rows from "../data/embeddings.json";
import vocab from "../data/vocab.json";
import Plot3D from "../components/Plot3D.jsx";
import { categoryColor } from "../lib/categories.js";
import { analogy, nearest, vecOf } from "../lib/vec.js";

const byWord = Object.fromEntries(rows.map((r) => [r.word, r]));
const fmt = (n) => (n >= 0 ? "+" : "") + n.toFixed(3);

// Sparse extra words (outside the core 50) are kept by the relaxed checker and
// show up here with category "other". Make them selectable + visible.
const EXTRAS = rows.filter((r) => r.category === "other").map((r) => r.word);
const GROUPS = { ...vocab.categories, ...(EXTRAS.length ? { other: EXTRAS } : {}) };
const PRESENT_CATEGORIES = Object.keys(GROUPS);

function WordSelect({ value, onChange }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="wsel">
      {Object.entries(GROUPS).map(([cat, toks]) => (
        <optgroup key={cat} label={cat}>
          {toks.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

export default function EmbeddingPage() {
  const [inspected, setInspected] = useState("apple");
  const [a, setA] = useState("king");
  const [b, setB] = useState("man");
  const [c, setC] = useState("woman");
  const [showAnalogy, setShowAnalogy] = useState(false);

  const neighbors = useMemo(
    () => nearest(rows, vecOf(byWord[inspected]), { exclude: [inspected], k: 5 }),
    [inspected]
  );

  const ana = useMemo(() => analogy(rows, a, b, c, 5), [a, b, c]);

  // Highlight set + extra plot traces (the analogy parallelogram) when active.
  const { highlight, extras } = useMemo(() => {
    if (!showAnalogy) return { highlight: new Set([inspected]), extras: [] };
    const top = ana.results[0]?.word;
    const hl = new Set([a, b, c, top].filter(Boolean));
    const pa = vecOf(byWord[a]);
    const pb = vecOf(byWord[b]);
    const pc = vecOf(byWord[c]);
    const t = ana.target;
    const line = (p, q, color, dash) => ({
      type: "scatter3d",
      mode: "lines",
      x: [p[0], q[0]],
      y: [p[1], q[1]],
      z: [p[2], q[2]],
      line: { color, width: 5, dash },
      hoverinfo: "skip",
    });
    const target = {
      type: "scatter3d",
      mode: "markers+text",
      x: [t[0]],
      y: [t[1]],
      z: [t[2]],
      text: ["= result"],
      textposition: "bottom center",
      marker: { size: 9, color: "#f7768e", symbol: "diamond" },
      hovertemplate: "result vector<br>%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>",
    };
    // Parallelogram: (b -> a) is the same offset as (c -> target).
    return {
      highlight: hl,
      extras: [
        line(pb, pa, "#6ea8fe", "solid"), // the "b -> a" difference
        line(pc, t, "#f7768e", "dot"), // same offset applied at c
        target,
      ],
    };
  }, [showAnalogy, inspected, a, b, c, ana]);

  return (
    <div>
      <h2>Embeddings in 3D</h2>
      <p className="muted" style={{ maxWidth: 820 }}>
        Each of the {rows.length} tokens is a point in 3-dimensional space —
        plotted directly, no dimensionality reduction. Words used in similar
        ways end up near each other. Rotate with the mouse, click a point to
        inspect it, or try word arithmetic below.{" "}
        <span style={{ color: "var(--muted)" }}>
          (In just 3 dimensions, analogies are approximate — the answer is
          usually in the top few neighbors.)
        </span>
      </p>

      <div className="embed-grid">
        <div className="card plot-card">
          <Plot3D
            rows={rows}
            highlight={highlight}
            extras={extras}
            onClickWord={(w) => {
              setInspected(w);
              setShowAnalogy(false);
            }}
          />
          <div className="legend">
            {PRESENT_CATEGORIES.map((cat) => (
              <span key={cat} className="legend-item">
                <span className="dot" style={{ background: categoryColor(cat) }} />
                {cat}
              </span>
            ))}
          </div>
        </div>

        <div className="side">
          {/* Vector inspector */}
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Inspect a word</h3>
            <WordSelect value={inspected} onChange={(w) => { setInspected(w); setShowAnalogy(false); }} />
            <div className="vec-row">
              <span className="dot" style={{ background: categoryColor(byWord[inspected].category) }} />
              <code>{inspected}</code>
              <span className="muted" style={{ marginLeft: "auto" }}>
                ({fmt(byWord[inspected].x)}, {fmt(byWord[inspected].y)}, {fmt(byWord[inspected].z)})
              </span>
            </div>
            <div className="muted" style={{ fontSize: 12, margin: "12px 0 6px" }}>
              Nearest neighbors (cosine)
            </div>
            {neighbors.map((n) => (
              <div key={n.word} className="nbr">
                <span className="dot" style={{ background: categoryColor(n.category) }} />
                <code>{n.word}</code>
                <span className="freq-bar" style={{ margin: "0 8px" }}>
                  <span
                    className="freq-fill"
                    style={{ width: `${Math.max(0, n.score) * 100}%`, background: categoryColor(n.category) }}
                  />
                </span>
                <span className="muted mono">{n.score.toFixed(2)}</span>
              </div>
            ))}
          </div>

          {/* Analogy tool */}
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Word arithmetic</h3>
            <div className="analogy-row">
              <WordSelect value={a} onChange={setA} />
              <span>−</span>
              <WordSelect value={b} onChange={setB} />
              <span>+</span>
              <WordSelect value={c} onChange={setC} />
            </div>
            <div className="presets">
              {vocab.analogies.map((p) => (
                <button
                  key={`${p.a}-${p.b}-${p.c}`}
                  className="preset"
                  onClick={() => { setA(p.a); setB(p.b); setC(p.c); setShowAnalogy(true); }}
                >
                  {p.a} − {p.b} + {p.c}
                </button>
              ))}
            </div>
            <button className="primary" onClick={() => setShowAnalogy(true)}>
              Compute & show in 3D
            </button>
            {showAnalogy && (
              <div style={{ marginTop: 12 }}>
                <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
                  <code>{a} − {b} + {c}</code> ≈
                </div>
                {ana.results.map((r, i) => (
                  <div key={r.word} className="nbr">
                    <span className="dot" style={{ background: categoryColor(r.category) }} />
                    <code style={{ fontWeight: i === 0 ? 700 : 400 }}>{r.word}</code>
                    {i === 0 && <span className="badge">top</span>}
                    <span className="freq-bar" style={{ margin: "0 8px" }}>
                      <span
                        className="freq-fill"
                        style={{ width: `${Math.max(0, r.score) * 100}%`, background: categoryColor(r.category) }}
                      />
                    </span>
                    <span className="muted mono">{r.score.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
