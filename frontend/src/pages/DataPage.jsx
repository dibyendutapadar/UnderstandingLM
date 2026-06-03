import vocab from "../data/vocab.json";
import stats from "../data/corpus_stats.json";
import StatCard from "../components/StatCard.jsx";
import { categoryColor } from "../lib/categories.js";

const maxFreq = Math.max(...Object.values(stats.token_freq), 1);

function TokenChip({ token, cat }) {
  const freq = stats.token_freq[token] ?? 0;
  const pct = Math.round((freq / maxFreq) * 100);
  return (
    <div className="token-chip" title={`appears ${freq.toLocaleString()} times`}>
      <span className="dot" style={{ background: categoryColor(cat) }} />
      <span className="tok">{token}</span>
      <span className="freq-bar">
        <span className="freq-fill" style={{ width: `${pct}%`, background: categoryColor(cat) }} />
      </span>
      <span className="freq-num muted">{freq.toLocaleString()}</span>
    </div>
  );
}

export default function DataPage() {
  return (
    <div>
      <h2>The Language</h2>
      <p className="muted" style={{ maxWidth: 760 }}>
        A {vocab.size}-token vocabulary with a hand-designed grammar. Sentences
        are generated (LLM or templates), then a checker rejects anything
        containing a token outside the vocabulary — so the corpus is guaranteed
        to use <em>only</em> these {vocab.counts.words} words and{" "}
        {vocab.counts.punctuation} punctuation marks.
      </p>

      <div className="stat-grid" style={{ marginTop: 20 }}>
        <StatCard value={vocab.size} label="Tokens" />
        <StatCard value={vocab.counts.words} label="Words" />
        <StatCard value={vocab.counts.punctuation} label="Punctuation" />
        <StatCard value={stats.num_sentences.toLocaleString()} label="Sentences" />
        <StatCard value={stats.num_tokens_emitted.toLocaleString()} label="Tokens used" />
        <StatCard value={`${stats.vocab_covered}/${stats.vocab_size}`} label="Coverage" />
        <StatCard value={`${(stats.rejection_rate * 100).toFixed(1)}%`} label="Rejected" />
      </div>

      <h3 style={{ marginTop: 36 }}>Vocabulary by category</h3>
      <p className="muted" style={{ marginTop: 0 }}>
        Bar = how often each token appears in the corpus.
      </p>
      <div className="category-cols">
        {Object.entries(vocab.categories).map(([cat, tokens]) => (
          <div className="card cat-card" key={cat}>
            <div className="cat-head">
              <span className="dot" style={{ background: categoryColor(cat) }} />
              {cat}
              <span className="muted" style={{ marginLeft: "auto", fontSize: 12 }}>
                {stats.category_freq[cat]?.toLocaleString()}
              </span>
            </div>
            {tokens.map((t) => (
              <TokenChip key={t} token={t} cat={cat} />
            ))}
          </div>
        ))}
      </div>

      <h3 style={{ marginTop: 36 }}>
        Sparse extra words{" "}
        <span className="muted" style={{ fontSize: 13, fontWeight: 400 }}>
          (outside the core {vocab.size})
        </span>
      </h3>
      <p className="muted" style={{ marginTop: 0, maxWidth: 760 }}>
        The relaxed checker keeps real-English words the model slips in (as long
        as a sentence has at least one core token). They enter the embedding as a
        sparse tail — concept words like <code>fruit</code> or <code>animal</code>{" "}
        tend to settle near the things they describe.
      </p>
      {stats.num_extra_types > 0 ? (
        <>
          <div className="stat-grid" style={{ marginBottom: 14 }}>
            <StatCard value={stats.num_extra_types} label="Extra word types" />
            <StatCard value={stats.num_extra_tokens.toLocaleString()} label="Extra tokens" />
            <StatCard value={stats.sentences_with_extra.toLocaleString()} label="Sentences w/ extra" />
          </div>
          <div className="card">
            <div className="extra-chips">
              {Object.entries(stats.extra_freq).map(([w, c]) => (
                <span key={w} className="extra-chip" title={`appears ${c} times`}>
                  <span className="dot" style={{ background: categoryColor("other") }} />
                  <code>{w}</code>
                  <span className="muted">{c}</span>
                </span>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="notice">
          None in this corpus — it was generated with the strict templates. Extras
          appear when you generate with OpenAI (the relaxed checker keeps them).
        </div>
      )}

      <h3 style={{ marginTop: 36 }}>Sample sentences</h3>
      <div className="card">
        <div className="samples">
          {stats.samples.map((s, i) => (
            <code key={i} className="sample">
              {s}
            </code>
          ))}
        </div>
      </div>

      {stats.rejection_examples?.length > 0 && (
        <>
          <h3 style={{ marginTop: 36 }}>Rejected by the checker</h3>
          <div className="card">
            {stats.rejection_examples.map((ex, i) => (
              <div key={i} className="sample">
                <code>{ex.text}</code>{" "}
                <span className="muted">— unknown: {ex.unknown.join(", ")}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
