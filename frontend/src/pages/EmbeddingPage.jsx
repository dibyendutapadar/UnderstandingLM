export default function EmbeddingPage() {
  return (
    <div>
      <h2>Embeddings in 3D</h2>
      <p className="muted" style={{ maxWidth: 720 }}>
        Every word becomes a point in 3-dimensional space. This page will let
        you rotate the cloud, select words, read their (x, y, z) vectors, and
        try word arithmetic like <code>king − man + woman ≈ queen</code>.
      </p>
      <div className="notice" style={{ marginTop: 24 }}>
        Placeholder — wired up in Step 3 once embeddings are trained.
      </div>
    </div>
  );
}
