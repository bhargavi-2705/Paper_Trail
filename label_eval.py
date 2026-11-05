"""Interactive labelling for HAND-WRITTEN eval questions.

eval/questions.jsonl ships with real questions but empty gold_pages. This tool
retrieves candidates for each unlabelled question and asks you which ones
actually answer it; you can also type a page number directly. Your judgement
is the gold label -- that's the point.

    python scripts/label_eval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config    # noqa: E402
from src.retrieve import Retriever    # noqa: E402

PATH = ROOT / "eval" / "questions.jsonl"


def main():
    rows = [json.loads(l) for l in PATH.read_text().splitlines() if l.strip()]
    todo = [r for r in rows if not r.get("gold_pages")]
    if not todo:
        print("Everything is labelled.")
        return
    retriever = Retriever(load_config())
    print(f"{len(todo)} unlabelled. Enter = skip, 'q' = save and quit.\n"
          "Choose candidate numbers (e.g. '1 3'), or type 'p 12' for page 12 directly.\n")

    for r in todo:
        print("=" * 70)
        print(f"{r['id']}  {r['question']}")
        if r.get("gold_doc"):
            print(f"(expected in: {r['gold_doc']})")
        hits = retriever.search(r["question"], k=8)
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h['doc']} p.{h['page']}  {h['text'][:160].strip()}...")
        ans = input("> ").strip()
        if ans.lower() == "q":
            break
        if not ans:
            continue
        if ans.lower().startswith("p "):
            pages = [int(x) for x in ans[2:].split() if x.isdigit()]
            doc = r.get("gold_doc") or (hits[0]["doc"] if hits else None)
        else:
            picked = [hits[int(x) - 1] for x in ans.split() if x.isdigit() and 0 < int(x) <= len(hits)]
            pages = sorted({h["page"] for h in picked})
            doc = picked[0]["doc"] if picked else r.get("gold_doc")
        if pages:
            r["gold_pages"], r["gold_doc"] = pages, doc
            print(f"  -> {doc} pages {pages}")

    PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n = sum(1 for r in rows if r.get("gold_pages"))
    print(f"\nsaved. {n}/{len(rows)} labelled.")


if __name__ == "__main__":
    main()
