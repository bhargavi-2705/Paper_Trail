"""Compare two saved generation runs on the questions they share.

    python scripts/compare_runs.py baseline_gen agent_gen

Only faithfulness and answer latency are compared: hit@k / MRR score the plain
retriever in run_eval.py, so they are identical for both runs by construction.
What this does NOT measure: answer completeness or correctness, only whether the
answer is grounded in the evidence it was given. The agent also sees up to 10
excerpts per answer against the baseline's 5, so the contexts differ.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(tag):
    data = json.loads((ROOT / "eval" / "results" / f"{tag}.json").read_text())
    return {r["id"]: r for r in data["per_question"]}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/compare_runs.py <baseline_tag> <other_tag>")
    ta, tb = sys.argv[1], sys.argv[2]
    a, b = load(ta), load(tb)
    ids = sorted(set(a) & set(b))
    if not ids:
        sys.exit("The two runs share no question ids.")
    both = [i for i in ids if a[i].get("faithfulness") is not None and b[i].get("faithfulness") is not None]

    print(f"shared questions: {len(ids)}   scored in both: {len(both)}\n")
    fa, fb = mean(a[i]["faithfulness"] for i in both), mean(b[i]["faithfulness"] for i in both)
    la = mean(a[i]["answer_latency_s"] for i in ids if a[i].get("answer_latency_s"))
    lb = mean(b[i]["answer_latency_s"] for i in ids if b[i].get("answer_latency_s"))
    print(f"{'metric':<26}{ta:>14}{tb:>14}{'delta':>10}")
    if fa is not None:
        print(f"{'faithfulness (mean)':<26}{fa:>14.3f}{fb:>14.3f}{fb - fa:>+10.3f}")
    if la and lb:
        print(f"{'answer latency (s)':<26}{la:>14.2f}{lb:>14.2f}{lb / la:>9.1f}x")

    wins = [i for i in both if b[i]["faithfulness"] > a[i]["faithfulness"] + 1e-9]
    loss = [i for i in both if b[i]["faithfulness"] < a[i]["faithfulness"] - 1e-9]
    ties = len(both) - len(wins) - len(loss)
    print(f"\nper question, {tb} vs {ta}: better on {len(wins)}, worse on {len(loss)}, same on {ties}")
    for i in loss:
        print(f"  worse  {i}: {a[i]['faithfulness']:.2f} -> {b[i]['faithfulness']:.2f}")
    for i in wins:
        print(f"  better {i}: {a[i]['faithfulness']:.2f} -> {b[i]['faithfulness']:.2f}")
    print(f"\nWith n={len(both)}, one question moves the mean by {1 / max(len(both), 1):.3f} at most; "
          "treat small gaps as noise.")


if __name__ == "__main__":
    main()
