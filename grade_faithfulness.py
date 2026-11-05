"""Hand-grade a sample of saved answers, then compare your grades with the LLM judge.

    python scripts/grade_faithfulness.py              # grade ~15 answers, saves as you go
    python scripts/grade_faithfulness.py --report     # just print agreement / kappa so far

Uses answers already saved by run_eval.py, so it makes NO API calls.
You see the question, the answer, and the exact excerpts the answerer saw. You do
NOT see the judge's score (blind grading, so it can't anchor you).

  1 = every factual claim in the answer is supported by the excerpts shown
  0 = at least one claim is NOT in the excerpts (even if it happens to be true)

The question is "is it in the excerpts", not "is it true" -- you don't need to
have read the papers.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config      # noqa: E402
from src.judge import validate_judge    # noqa: E402
from src.retrieve import Retriever      # noqa: E402

RESULTS = ROOT / "eval" / "results"
STRICT = 0.999   # judge says "faithful" only if every claim was supported


def pick_sample(per_q, n_perfect, seed):
    """All answers the judge scored below 1.0, plus a random set of 'perfect' ones.
    The perfect ones are what matter most: a lenient judge hides in them."""
    scored = [r for r in per_q if r.get("faithfulness") is not None and r.get("answer")]
    low = [r for r in scored if r["faithfulness"] < STRICT]
    perfect = [r for r in scored if r["faithfulness"] >= STRICT]
    rng = random.Random(seed)
    rng.shuffle(perfect)
    sample = low + perfect[:n_perfect]
    rng.shuffle(sample)          # grade in random order
    return sample


def report(grades):
    recs = [g for g in grades.values() if g.get("human_label") is not None]
    if not recs:
        print("No grades yet.")
        return
    stats = validate_judge(
        [{"judge_score": g["judge_score"], "human_label": g["human_label"]} for g in recs],
        threshold=STRICT)
    jf = lambda g: g["judge_score"] >= STRICT
    both_ok = sum(jf(g) and g["human_label"] == 1 for g in recs)
    lenient = sum(jf(g) and g["human_label"] == 0 for g in recs)     # judge OK'd, you didn't
    strict_ = sum((not jf(g)) and g["human_label"] == 1 for g in recs)  # judge flagged, you didn't
    both_bad = sum((not jf(g)) and g["human_label"] == 0 for g in recs)
    kappa = stats["kappa"]
    print(f"\n== judge vs your grades (n={stats['n']}) ==")
    print(f"  agreement                    {stats['agreement']:.3f}")
    print(f"  Cohen's kappa                {kappa:.3f}" if kappa is not None else "  Cohen's kappa                n/a")
    print(f"  both say faithful            {both_ok}")
    print(f"  judge faithful, you not      {lenient}   <- judge too lenient")
    print(f"  judge flagged, you faithful  {strict_}   <- judge too strict")
    print(f"  both say unfaithful          {both_bad}")
    print("  (small n: quote these with the sample size, the error bars are wide)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline_gen", help="which saved run to grade")
    ap.add_argument("--n-perfect", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    grades_path = RESULTS / f"judge_validation_{args.tag}.json"
    grades = json.loads(grades_path.read_text()) if grades_path.exists() else {}
    if args.report:
        return report(grades)

    run = json.loads((RESULTS / f"{args.tag}.json").read_text())
    sample = pick_sample(run["per_question"], args.n_perfect, args.seed)
    todo = [r for r in sample if grades.get(r["id"], {}).get("human_label") is None]
    print(f"{len(sample)} answers in the sample, {len(todo)} left to grade. "
          "Enter 1 / 0 (s = skip, q = save and quit).\n")

    retriever = Retriever(load_config(args.config))
    for r in todo:
        hits = retriever.search(r["question"])
        print("=" * 78)
        print(f"{r['id']}  QUESTION: {r['question']}\n")
        print(f"ANSWER:\n{r['answer']}\n")
        print("EXCERPTS THE ANSWER WAS WRITTEN FROM:")
        for i, h in enumerate(hits, 1):
            print(f"\n[{i}] {h['doc']} p.{h['page']}\n{h['text']}")
        while True:
            v = input("\nevery claim supported by the excerpts? 1 / 0 > ").strip().lower()
            if v in ("1", "0", "s", "q"):
                break
        if v == "q":
            break
        if v == "s":
            continue
        grades[r["id"]] = {"judge_score": r["faithfulness"], "human_label": int(v)}
        grades_path.parent.mkdir(parents=True, exist_ok=True)
        grades_path.write_text(json.dumps(grades, indent=2))
    report(grades)


if __name__ == "__main__":
    main()
