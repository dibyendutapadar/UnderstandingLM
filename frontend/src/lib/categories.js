// Consistent color per token category, reused by the Data page and the 3D
// embedding viewer so a word is the same color everywhere.
export const CATEGORY_COLORS = {
  people: "#6ea8fe",
  fruit: "#f7768e",
  color: "#ffd866",
  animal: "#9ece6a",
  sentiment: "#bb9af7",
  size: "#7dcfff",
  verb: "#ff9e64",
  function: "#737aa2",
  place: "#73daca",
  time: "#e0af68",
  punctuation: "#565f89",
};

export function categoryColor(cat) {
  return CATEGORY_COLORS[cat] || "#888";
}
