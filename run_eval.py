"""Retrieval + faithfulness evaluation.

    python scripts/run_eval.py                      # retrieval only (no API calls)
    python scripts/run_eval.py --generation         # + answers + LLM-judge faithfulness
    python scripts/run_eval.py --generation --pause 20   # wait 20s between questions (rate limits)
    python scripts/run_eval.py --generation --agent --stride 3   # agent on every 3rd question
    python scripts/run_eval.py --split human        # only hand-written questions
    python scripts/run_eval.py --tag baseline       # label the run in eval/results/

Note: hit@k / MRR always score the plain retriever. They do not change with --agent;
only faithfulness and latency do.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                      # noqa: E402
from src.metrics import aggregate, score_question       # noqa: E402
from src.retrieve import Retriever                      # noqa: E402

KS = (1, 3, 5, 10)


def load_questions(path, split=None):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("gold_pages")]
    if split:
        rows = [r for r in rows if r.get("split", "human") == split]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--split", default=None, help="human | synthetic")
    ap.add_argument("--generation", action="store_true", help="also score answer faithfulness")
    ap.add_argument("--page-tol", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=0, help="seconds to wait between questions")
    ap.add_argument("--agent", action="store_true", help="use the LangGraph agent, not the baseline chain")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth question (spreads a subset across papers)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.agent:
        cfg["agent"]["enabled"] = True
    questions = load_questions(args.questions, args.split)
    if args.stride > 1:
        questions = questions[::args.stride]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        sys.exit(f"No labelled questions in {args.questions} (need gold_pages).")

    retriever = Retriever(cfg)
    llm = None
    if args.generation:
        from src.llm import get_llm
        llm = get_llm(cfg)

    per_q, latencies, stopped = [], [], None
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=max(KS))
        latencies.append(time.perf_counter() - t0)
        row = score_question(hits, q.get("gold_doc"), q["gold_pages"], KS, args.page_tol)
        row.update(id=q["id"], question=q["question"], split=q.get("split", "human"))

        if args.generation:
            try:
                from src.answer import answer
                from src.judge import faithfulness
                t1 = time.perf_counter()
                out = answer(q["question"], cfg, retriever, llm)
                row["answer_latency_s"] = time.perf_counter() - t1
                f = faithfulness(out["answer"], out["sources"], llm)
                row["faithfulness"] = f["score"]
                row["n_claims"] = len(f["claims"])
                row["answer"] = out["answer"]
                if out.get("trace"):
                    row["trace"] = out["trace"]
            except Exception as e:
                stopped = f"{q['id']}: {type(e).__name__}: {str(e)[:200]}"
                latencies.pop()        # keep latencies aligned with finished questions
                break
        per_q.append(row)
        print(f"  {row['id']:<6} hit@5={row['hit@5']:.0f} mrr={row['mrr']:.2f}"
              + (f" faith={row.get('faithfulness')}" if args.generation else ""))
        if args.generation and args.pause:
            time.sleep(args.pause)

    if stopped:
        print(f"\n!! stopped early at {stopped}")
        print(f"!! saving the {len(per_q)} finished questions (of {len(questions)})")
    if not per_q:
        sys.exit("No questions finished.")

    summary = aggregate(per_q, KS)
    summary["n_requested"] = len(questions)
    summary["retrieval_latency_s_mean"] = sum(latencies) / len(latencies)
    if args.generation:
        fs = [r["faithfulness"] for r in per_q if r.get("faithfulness") is not None]
        summary["faithfulness_mean"] = sum(fs) / len(fs) if fs else None
        summary["faithfulness_n_scored"] = len(fs)
        al = [r["answer_latency_s"] for r in per_q if r.get("answer_latency_s")]
        summary["answer_latency_s_mean"] = sum(al) / len(al) if al else None

    cfg_snapshot = {"retrieval": cfg["retrieval"], "chunking": cfg["chunking"],
                    "agent": cfg["agent"], "embedding": cfg["embedding"],
                    "llm": {k: v for k, v in cfg["llm"].items() if k != "base_url"}}
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(
        {"tag": args.tag, "config": cfg_snapshot, "summary": summary, "per_question": per_q},
        indent=2))

    print("\n== summary ==")
    for k, v in summary.items():
        print(f"  {k:<28} {v:.3f}" if isinstance(v, float) else f"  {k:<28} {v}")
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
