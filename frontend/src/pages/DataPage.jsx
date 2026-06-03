export default function DataPage() {
  return (
    <div>
      <h2>The Language</h2>
      <p className="muted" style={{ maxWidth: 720 }}>
        A 50-token vocabulary and a hand-designed grammar. This page will show
        the token list by category, corpus statistics (how often each word
        appears), and sample generated sentences.
      </p>
      <div className="notice" style={{ marginTop: 24 }}>
        Placeholder — wired up in Step 2 once the corpus is generated.
      </div>
    </div>
  );
}
