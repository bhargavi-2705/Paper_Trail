# notes-rag-kb — RAG over an ML paper knowledge base, with coverage analysis

A retrieval-augmented QA system over a corpus of machine-learning papers, built
**baseline-first**: every component (hybrid retrieval, reranking, the agent loop)
has to earn its place against a measured baseline on a hand-labelled eval set.

The project answers three questions most RAG demos skip:

1. **Does each component actually help?** → ablation table, one row per variant.
2. **How do you evaluate without gold answers?** → hand-labelled retrieval gold
   pages + LLM-judge faithfulness, with the judge validated against hand grades.
3. **What is *missing* from the knowledge base?** → topic coverage analysis with a
   calibrated gap threshold.

## Results

> Numbers go here after you run `python scripts/ablations.py`. Do not fill this in
> by hand — the table is generated into `eval/results/ablations.md`.

| variant | hit@1 | hit@3 | hit@5 | recall@5 | mrr | faithfulness | latency (s) |
|---|---|---|---|---|---|---|---|
| dense_only | | | | | | | |
| hybrid | | | | | | | |
| dense_rerank | | | | | | | |
| hybrid_rerank | | | | | | | |
| baseline_chain | | | | | | | |
| agent_full | | | | | | | |

## Quickstart

Use a virtual environment. This project pins LangChain 1.x; installing it into a
shared environment that holds other LLM tooling (crewai, embedchain, langchain-groq)
will produce version conflicts in both directions.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or requirements.lock.txt for exact versions

python scripts/fetch_corpus.py      # downloads 10 arXiv PDFs into data/raw/
python -m src.build_index           # chunk -> embed -> Chroma (first run downloads bge-small)

export ANTHROPIC_API_KEY=...        # or set llm.provider: openai in config.yaml
python -m src.answer "what problem does LoRA solve?"

streamlit run app.py                # demo UI
```

Then the evaluation loop:

```bash
python scripts/label_eval.py        # label gold pages for the 32 hand-written questions
python scripts/run_eval.py          # retrieval metrics, no API calls
python scripts/run_eval.py --generation   # + answers + faithfulness
python scripts/ablations.py         # full comparison table
python scripts/run_coverage.py      # topic gaps
python scripts/run_quiz.py --adaptive
pytest                              # 22 tests, no network needed
```

## Corpus

Ten typed (LaTeX-generated) arXiv papers covering transformers, retrieval,
fine-tuning, optimisation and in-context learning — see `corpus.yaml`. They are
fetched at run time and **never committed**; no handwritten or scanned material,
so no OCR path is needed.

## Pipeline

```
PDF (page-level)  ->  chunk (page-bounded)  ->  bge-small embeddings  ->  Chroma
                                                                            |
query ->  dense top-20  +  BM25 top-20  ->  RRF fusion  ->  CrossEncoder rerank -> top-5
                                                                            |
                             baseline chain  OR  LangGraph agent  ->  cited answer
                             (decompose -> retrieve -> critic -> re-query -> synthesize)
```

Every stage is a flag in `config.yaml`, which is what makes the ablations possible.

## Evaluation design

**Retrieval** is scored against hand-labelled gold pages: `hit@k`, `recall@k`,
`precision@5`, `MRR` (`src/metrics.py`, plain Python, unit-tested). Chunks are
page-bounded, so a gold label is a page number. `--page-tol` allows a ±1 page
match because a chunk at a page boundary can legitimately answer a question
labelled on the neighbouring page.

Two question splits, reported separately:

- **human** (`eval/questions.jsonl`) — 32 questions written by hand, gold pages
  assigned via `scripts/label_eval.py`. These are the headline numbers.
- **synthetic** (`scripts/generate_eval.py`) — questions generated *from* a chunk,
  so the gold page is known by construction. Useful for volume, but **optimistic**:
  a question written from a passage shares that passage's vocabulary. Never
  reported alone.

**Generation** is scored with LLM-judge faithfulness (share of answer claims
supported by the retrieved excerpts) — no gold answers required, which is what
makes it usable on a self-built corpus. The judge is then checked against hand
grades with `validate_judge()`, which reports raw agreement **and Cohen's kappa**.
Report those next to any faithfulness number.

**Coverage** probes each syllabus topic and flags thin ones. The threshold is
calibrated against topics known to be present and one seeded topic known to be
absent (`reinforcement-learning`), and the calibration accuracy is reported with
the result.

## Limitations

- The eval set is small (32 hand-written questions). Differences of a few points
  between variants are inside the noise; only large gaps are meaningful.
- Judge validation rests on a few dozen hand grades, so agreement and kappa have
  wide error bars.
- Gold pages are single-annotator (mine). No inter-annotator agreement.
- Page-bounded chunks guarantee exact citations but can split an idea that runs
  across a page break. The `per_page` and chunk-size ablations measure the cost.
- Coverage gaps are detected by retrieval score, which reflects *vocabulary*
  overlap as much as conceptual coverage. The LLM check (`--llm`) is a partial
  mitigation, not a fix.
- Faithfulness measures grounding in retrieved text, not truth. A confidently
  wrong passage produces a faithful, wrong answer.

## Layout

```
src/     ingest chunk embed vectorstore retrieve answer agent metrics judge coverage quiz
scripts/ fetch_corpus build_index run_eval ablations generate_eval label_eval run_coverage run_quiz
eval/    questions.jsonl + results/
tests/   22 unit tests (metrics, chunking, RRF, judge, calibration, agent graph)
docs/    DECISIONS.md — one entry per experiment, with numbers
```
