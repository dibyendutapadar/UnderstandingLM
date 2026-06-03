// A small numeric matrix shown as a table. Each cell is tinted by its value
// (diverging blue/red around zero) so structure is visible at a glance.
function cellColor(v, scale) {
  if (scale === 0) return "transparent";
  const t = Math.max(-1, Math.min(1, v / scale));
  // blue for negative, red for positive, transparent near zero
  const a = Math.abs(t) * 0.55;
  return t >= 0 ? `rgba(247,118,142,${a})` : `rgba(110,168,254,${a})`;
}

export default function MatrixTable({ rows, rowLabels, colLabels, title, hint }) {
  const scale = Math.max(1e-6, ...rows.flat().map((v) => Math.abs(v)));
  return (
    <div className="card matrix-card">
      <div className="matrix-head">
        <span className="matrix-title">{title}</span>
        {hint && <span className="muted matrix-hint">{hint}</span>}
        <span className="muted matrix-shape">
          {rows.length}×{rows[0]?.length ?? 0}
        </span>
      </div>
      <div className="matrix-scroll">
        <table className="matrix">
          <thead>
            <tr>
              <th></th>
              {(colLabels ?? rows[0].map((_, i) => i)).map((c, i) => (
                <th key={i}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                <th className="rowlab">{rowLabels?.[i] ?? i}</th>
                {row.map((v, j) => (
                  <td key={j} style={{ background: cellColor(v, scale) }}>
                    {v.toFixed(2)}
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
