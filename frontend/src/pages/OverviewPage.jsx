const PIPELINE = [
  "Tokens",
  "Embedding (3D)",
  "Q / K / V",
  "Attention",
  "MLP",
  "Logits",
  "Next token",
];

export default function OverviewPage() {
  return (
    <div>
      <h2>Visualize & Understand Language Models</h2>
      <p className="muted" style={{ maxWidth: 720 }}>
        Real language models are impossible to picture: thousands of dimensions,
        billions of numbers. This project shrinks everything until it fits on
        screen — a <strong>50-word language</strong> and a{" "}
        <strong>3-dimensional embedding space</strong> you can rotate with your
        mouse. Because we invent the grammar ourselves, we know the ground-truth
        rules and can check whether the model actually learns them.
      </p>

      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>The pipeline we'll make visible</h3>
        <div className="pipeline">
          {PIPELINE.map((p, i) => (
            <span key={p} style={{ display: "contents" }}>
              <span className="box">{p}</span>
              {i < PIPELINE.length - 1 && <span className="arrow">→</span>}
            </span>
          ))}
        </div>
        <p className="muted" style={{ marginBottom: 0, marginTop: 16 }}>
          We build it one stage at a time. Use the nav on the left.
        </p>
      </div>

      <div className="notice" style={{ marginTop: 24 }}>
        Step 1 (skeleton) is live. Next: generate the tiny language, then train
        and explore the 3D embeddings.
      </div>
    </div>
  );
}
