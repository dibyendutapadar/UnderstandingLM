import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { categoryColor } from "../lib/categories.js";

// Use the slim plotly build instead of react-plotly.js's default full bundle.
const Plot = createPlotlyComponent(Plotly);

const DARK = {
  paper_bgcolor: "#0f1117",
  plot_bgcolor: "#0f1117",
  font: { color: "#e6e8ee" },
};

/**
 * 3D scatter of word embeddings.
 *  - rows: [{word, x, y, z, category}]
 *  - highlight: Set of words to emphasize (others dimmed). Empty = all bright.
 *  - extras: optional extra traces (e.g. an analogy result point + arrow)
 *  - onClickWord: callback(word)
 */
export default function Plot3D({ rows, highlight, extras = [], onClickWord, height = 560 }) {
  const hl = highlight && highlight.size > 0 ? highlight : null;

  const trace = {
    type: "scatter3d",
    mode: "markers+text",
    x: rows.map((r) => r.x),
    y: rows.map((r) => r.y),
    z: rows.map((r) => r.z),
    text: rows.map((r) => r.word),
    textposition: "top center",
    textfont: {
      size: rows.map((r) => (!hl || hl.has(r.word) ? 12 : 9)),
      color: rows.map((r) => (!hl || hl.has(r.word) ? "#e6e8ee" : "#5b6172")),
    },
    marker: {
      size: rows.map((r) => (!hl || hl.has(r.word) ? 6 : 4)),
      color: rows.map((r) => categoryColor(r.category)),
      opacity: 1,
      line: { width: 0 },
    },
    customdata: rows.map((r) => r.category),
    hovertemplate: "<b>%{text}</b> (%{customdata})<br>%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>",
  };

  // Dim non-highlighted points by lowering opacity via a second pass is awkward
  // in one trace; instead encode opacity through marker.color alpha is also
  // awkward, so we keep full color but shrink dimmed markers (above).

  return (
    <Plot
      data={[trace, ...extras]}
      layout={{
        ...DARK,
        height,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        showlegend: false,
        scene: {
          xaxis: { title: "x", gridcolor: "#2a2f3d", zerolinecolor: "#3a4152" },
          yaxis: { title: "y", gridcolor: "#2a2f3d", zerolinecolor: "#3a4152" },
          zaxis: { title: "z", gridcolor: "#2a2f3d", zerolinecolor: "#3a4152" },
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      onClick={(e) => {
        const p = e?.points?.[0];
        if (p && onClickWord && p.text) onClickWord(p.text);
      }}
    />
  );
}
