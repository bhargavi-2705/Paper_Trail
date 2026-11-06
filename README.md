# PaperTrail: RAG over ML papers, with retrieval ablations and coverage analysis

A retrieval-augmented QA system over 10 arXiv papers (transformers, retrieval, fine-tuning,
optimisation) that answers with page-level citations. It was built baseline-first: every component
had to beat the previous one on a labelled eval set, and the ones that didn't were switched off.
This README reports what was measured, including what didn't work.

**Corpus and index:** 10 papers, 245 pages, 909 chunks (1200 characters, 200 overlap).

## Results

**Retrieval** (32 labelled questions, strict page match, one question = 3.1 points)

| variant | hit@1 | hit@3 | hit@5 | MRR | latency (s) |
|---|---|---|---|---|---|
| dense only | 0.438 | 0.688 | 0.844 | 0.590 | 0.06 |
| dense + BM25 (RRF) | 0.406 | 0.719 | 0.750 | 0.579 | 0.07 |
| **dense + reranker (final)** | 0.438 | **0.844** | **0.969** | **0.649** | 0.39 |
| dense + BM25 + reranker | 0.438 | 0.812 | 0.969 | 0.645 | 0.40 |

Chunk size, with BM25 + reranker on (baseline: 1200 chars)

| chunking | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| 600 chars | 0.469 | 0.688 | 0.875 | 0.618 |
| 1200 chars (baseline) | 0.438 | 0.812 | 0.969 | 0.645 |
| 2000 chars | 0.438 | 0.750 | 0.844 | 0.615 |
| one chunk per page | 0.531 | 0.812 | 0.906 | 0.687 |

- **The reranker is the one clear win:** hit@3 0.69 to 0.84 and hit@5 0.84 to 0.97, for about 0.3 s extra per query.
- **BM25 added nothing once a reranker was on**, and slightly hurt without one, so it is switched off.
  Most questions are paraphrased concepts, which is where lexical search is weakest.
- **The reranker did not move hit@1** (0.438 either way): it gets the right page into the top 3 to 5
  but doesn't reliably put it first.
- **Chunking is a trade-off, not a win.** One chunk per page has the best hit@1 and MRR but lower
  hit@5 than 1200-char chunks. Gaps of one or two questions are noise at this sample size.

**Generation** (gpt-oss-20b via Groq, 32 questions, the same model writes and judges)

- LLM-judged faithfulness: **0.915 mean**. 26 of 32 answers scored 1.0, and the lowest was 0.0.
- Mean answer latency: 2.3 s.
- The judge has **not** been validated against human grades (see Limitations).

**Coverage analysis** (does the corpus cover a topic?)

- A first version failed its own test: AUC 0.40 on 6 labelled topics, and its 0.83 "accuracy" equalled
  the guess-everything-present baseline.
- After rewriting topic queries as natural-language phrases, **AUC 1.00 on 21 labelled topics**
  (15 known-present, 6 known-absent controls). The highest absent topic scored -3.7 and the lowest
  present topic 0.02.
- Only one real syllabus topic is flagged as thin (`evaluation-metrics`).

Every experiment, including the failures, is written up in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Pipeline

```
PDF (page level) -> chunk (stays inside a page) -> bge-small embeddings -> Chroma
query -> dense top-20 -> CrossEncoder rerank -> top-5 -> cited answer (page-level sources)
```

Every stage is a flag in `config.yaml`, which is how the ablations run. Also in the repo and
**not evaluated or exercised in the reported results**: a LangGraph agent loop (`src/agent.py`,
unit-tested with a fake LLM, off by default), a quiz generator, and a Streamlit demo (`app.py`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # or requirements.lock.txt for exact versions

python scripts/fetch_corpus.py             # downloads the 10 PDFs from arXiv (not committed)
python -m src.build_index                  # chunk, embed, store (first run downloads bge-small)

export OPENAI_API_KEY=...                  # any OpenAI-compatible provider (see below)
python -m src.answer "Why does LoRA add no inference latency?"
```

**LLM used for the reported results** (`config.yaml`, free tier on Groq):

```yaml
llm:
  provider: openai
  model: openai/gpt-oss-20b
  base_url: https://api.groq.com/openai/v1
  max_tokens: 2000
  temperature: 0
  reasoning_effort: low
```

Evaluation:

```bash
python scripts/label_eval.py               # assign gold pages to eval/questions.jsonl
python scripts/run_eval.py                 # retrieval metrics, no API calls
python scripts/ablations.py                # retrieval ablation table (~15 min)
python -m src.build_index                  # re-run afterwards: ablations leave the index in the last chunking mode
python scripts/run_eval.py --generation --pause 10 --tag baseline_gen   # answers + faithfulness
python scripts/run_coverage.py             # topic coverage
pytest                                     # unit tests, no network needed
```

## Evaluation design

- **Retrieval:** hit@k, recall@k, precision@5 and MRR against gold pages (`src/metrics.py`, plain
  Python, unit-tested). Matching is strict: the chunk must come from the labelled page. An earlier
  run allowed a neighbouring page and reported hit@5 = 1.000, which hid real misses.
- **Faithfulness:** an LLM judge splits each answer into claims and checks each against the retrieved
  excerpts. No gold answers are needed. It measures grounding, not truth.
- **Coverage:** score-based gap detection, tested against topics with known status.

## Limitations

- **32 questions, one annotator.** One question moves a metric by 3.1 points, so only large gaps
  (the reranker) are meaningful.
- **The faithfulness judge is unvalidated and judges its own output.** The same model writes and
  scores the answers, which tends to inflate scores. Faithfulness is groundedness, not correctness.
- **Page-bounded chunks can split an idea across a page break.** Question h010 missed for exactly this
  reason: the answer sentence runs from page 3 onto page 4, so neither chunk holds it.
- **Coverage is a small test.** The threshold was fitted and scored on the same 21 topics, and the
  queries were rewritten after the first attempt failed. The five newer control topics are the fairer test.
- **Free-tier LLM limits** (200K tokens/day) capped how much generation evaluation was run.

## Layout

```
src/      ingest chunk embed vectorstore retrieve answer agent metrics judge coverage quiz llm
scripts/  fetch_corpus build_index run_eval ablations label_eval run_coverage grade_faithfulness compare_runs
eval/     questions.jsonl, results/
tests/    unit tests (metrics, chunking, RRF, judge parsing, coverage, agent graph)
docs/     DECISIONS.md, INTERVIEW_NOTES.md
```
