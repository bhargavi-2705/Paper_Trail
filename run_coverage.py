"""Topic coverage analysis over the syllabus in corpus.yaml.

    python scripts/run_coverage.py            # retrieval scores only, no API calls
    python scripts/run_coverage.py --llm      # adds an LLM "does this substantively cover it?" verdict

corpus.yaml supplies:
  syllabus        topics to probe
  topic_queries   a natural-language query per topic (bare acronyms like 'peft' retrieve badly)
  known_present   topics YOU know the corpus covers  \  they test the detector: if it can't
  known_absent    topics YOU know it does not        /  separate these, it isn't usable
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                                         # noqa: E402
from src.coverage import (analyse, calibrate_threshold, probe_topic,       # noqa: E402
                          separability, verdict_accuracy)
from src.retrieve import Retriever                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "coverage.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = yaml.safe_load((ROOT / "corpus.yaml").read_text())
    syllabus = manifest["syllabus"]
    queries = manifest.get("topic_queries", {})
    present = set(manifest.get("known_present", []))
    absent = set(manifest.get("known_absent", []))
    retriever = Retriever(cfg)

    probes = [probe_topic(t, retriever, queries=queries) for t in syllabus]
    n_pos = sum(p["topic"] in present for p in probes)
    n_neg = sum(p["topic"] in absent for p in probes)
    n_lab = n_pos + n_neg
    majority = max(n_pos, n_neg) / n_lab if n_lab else None

    print(f"labelled topics: {n_pos} known-present, {n_neg} known-absent "
          f"(guessing 'everything present' scores {majority:.2f})\n" if n_lab else "no labelled topics\n")
    by_field = {}
    print(f"{'score':<14}{'AUC':>7}{'threshold':>11}{'accuracy':>10}")
    for field in ("best_score", "best_cosine"):
        thr, acc, n = calibrate_threshold(probes, present, absent, field)
        auc = separability(probes, present, absent, field)
        by_field[field] = {"threshold": thr, "accuracy": acc, "n_labelled": n, "auc": auc}
        if thr is not None:
            print(f"{field:<14}{auc:>7.2f}{thr:>11.3f}{acc:>10.2f}")
    field = max(by_field, key=lambda f: (by_field[f]["auc"] if by_field[f]["auc"] is not None else -1))
    cal = by_field[field]
    print(f"\nusing {field} for the gap flag (higher AUC)")

    llm = None
    if args.llm:
        from src.llm import get_llm
        llm = get_llm(cfg)

    report = analyse(syllabus, retriever, threshold=cal["threshold"], llm=llm, queries=queries, field=field)
    label = lambda t: "P" if t in present else ("A" if t in absent else "")

    print(f"\n{'topic':<28}{'known':>6}{'score':>9}{'docs':>6}{'gap':>6}" + ("   llm" if args.llm else ""))
    for r in sorted(report["topics"], key=lambda r: r["score_used"]):
        line = (f"{r['topic']:<28}{label(r['topic']):>6}{r['score_used']:>9.3f}{r['n_docs']:>6}"
                f"{'YES' if r['flagged_gap'] else '':>6}")
        if args.llm:
            line += f"   {r.get('llm_covers')}"
        print(line)
    print("  known: P = you know it's covered, A = you know it's absent")

    auc = cal["auc"]
    if auc is not None and auc < 0.9:
        print(f"\n!! The score does not reliably separate known-present from known-absent topics "
              f"(AUC {auc:.2f}; 0.5 = coin flip). Do not report gaps from the score alone.")
        fp = [p for p in probes if p["topic"] in absent and p[field] >= cal["threshold"]]
        for p in fp:
            print(f"\n   known-ABSENT '{p['topic']}' still scored {p[field]:.2f}. Top match:")
            h = p["hits"][0]
            print(f"   {h['doc']} p.{h['page']}: {h['text'][:260].strip()!r}")
    if args.llm:
        missing = [r["topic"] for r in report["topics"] if r.get("llm_covers") is None]
        if missing:
            print(f"\n!! {len(missing)} topics got no LLM verdict (quota or parse error), "
                  f"counted as wrong: {', '.join(missing[:6])}")
            print(f"   first error: {next(r['llm_why'] for r in report['topics'] if r.get('llm_covers') is None)}")
        acc, n = verdict_accuracy(report["topics"], present, absent)
        if acc is not None:
            print(f"\nLLM verdict matches what you know on {round(acc * n)} of {n} labelled topics ({acc:.2f}); "
                  f"the score-based flag gets {cal['accuracy']:.2f}")
            report["llm_verdict_accuracy"] = {"accuracy": acc, "n_labelled": n}

    gaps = [r["topic"] for r in report["topics"] if r["flagged_gap"]]
    print(f"\nflagged by score: {', '.join(gaps) if gaps else 'none'}")
    report["calibration"] = {**cal, "field": field, "by_field": by_field, "majority_baseline": majority,
                             "known_present": sorted(present), "known_absent": sorted(absent)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
