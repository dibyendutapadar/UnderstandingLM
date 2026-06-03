// Attention weights as a heatmap. Row = query token ("who is looking"),
// column = key token ("what it looks at"). Cell opacity ∝ weight.
export default function AttentionHeatmap({ attn, tokens }) {
  return (
    <div className="card matrix-card">
      <div className="matrix-head">
        <span className="matrix-title">Attention weights</span>
        <span className="muted matrix-hint">row attends to column (sums to 1)</span>
      </div>
      <div className="matrix-scroll">
        <table className="matrix heatmap">
          <thead>
            <tr>
              <th></th>
              {tokens.map((t, j) => (
                <th key={j}>{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {attn.map((row, i) => (
              <tr key={i}>
                <th className="rowlab">{tokens[i]}</th>
                {row.map((w, j) => (
                  <td
                    key={j}
                    title={`${tokens[i]} → ${tokens[j]}: ${w.toFixed(3)}`}
                    style={{
                      background: j <= i ? `rgba(126,231,135,${w})` : "transparent",
                      color: w > 0.6 ? "#0b1020" : "var(--text)",
                    }}
                  >
                    {j <= i ? w.toFixed(2) : ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
