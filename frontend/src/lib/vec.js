// Small vector helpers for the embedding viewer. Embeddings are 3D so this all
// stays trivially inspectable.

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a) => Math.sqrt(dot(a, a));

export function cosine(a, b) {
  const n = norm(a) * norm(b);
  return n === 0 ? 0 : dot(a, b) / n;
}

export const vecOf = (row) => [row.x, row.y, row.z];

// Top-k tokens by cosine similarity to a target vector.
export function nearest(rows, target, { exclude = [], k = 5 } = {}) {
  const ex = new Set(exclude);
  return rows
    .filter((r) => !ex.has(r.word))
    .map((r) => ({ word: r.word, category: r.category, score: cosine(vecOf(r), target) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// a - b + c, then nearest neighbors (excluding the three inputs).
export function analogy(rows, aWord, bWord, cWord, k = 5) {
  const byWord = Object.fromEntries(rows.map((r) => [r.word, vecOf(r)]));
  const target = add(sub(byWord[aWord], byWord[bWord]), byWord[cWord]);
  const results = nearest(rows, target, { exclude: [aWord, bWord, cWord], k });
  return { target, results };
}
