# Visualize & Understand Language Models

A teaching project (website + blog series) that makes language-model internals
**visible** by shrinking everything to a human-inspectable scale:

- a **50-token synthetic language** (we define the grammar, so we know the rules)
- a **3-dimensional embedding space** (3D plots directly — no PCA/t-SNE)
- a static, **in-browser** site where every number is inspectable

## Layout

```
backend/    Offline Python: corpus generation + model training. NOT hosted.
shared/     vocab.json — the contract shared by backend and frontend.
frontend/   React + Vite static site. This is what gets deployed.
```

The backend produces JSON artifacts (`embeddings.json`, `corpus_stats.json`)
that are copied into `frontend/src/data/`. The frontend does all math and
visualization client-side, so the whole thing hosts for free as static files.

## Build pipeline (run one step at a time)

All backend commands are run from the `backend/` directory.

```bash
# Step 1 — write the vocabulary contract
python vocab.py                       # -> shared/vocab.json (50 tokens)

# Step 2 — generate + validate the corpus
cp .env.example .env                  # add your OpenAI key
python generate_data.py --count 2000  # OpenAI short stories (rich co-occurrence)
#   ...or, with no API key / no spend:
python generate_data.py --dry-run --count 15000   # Python templates
python validate.py                    # split stories->sentences, vocab-check,
                                      # -> data/sentences.jsonl, corpus_stats.json

# Step 3 — train the 3D embeddings, then export to the frontend
python train_embeddings.py            # -> data/embeddings.json + analogy report
python export_to_frontend.py          # copies artifacts into frontend/src/data/

# Frontend
cd ../frontend && npm install && npm run dev
```

## Status

- [x] Step 1 — project skeleton + 50-token vocabulary
- [x] Step 2 — LLM/template corpus generation + checker
- [x] Step 3 — embedding training + 3D visualization page
- [ ] Step 4+ — Transformer Microscope (see `.claude/plans/`)
