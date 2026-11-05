"""Generate SYNTHETIC eval questions from indexed chunks.

Each question is written from one chunk, so its gold page is known by
construction -- no manual labelling. The catch, stated plainly in the README:
a question written *from* a chunk shares that chunk's vocabulary, so retrieval
scores on synthetic questions are optimistic. Keep them in a separate split and
always report human-written results next to them.

    python scripts/generate_eval.py --n 40 --out eval/questions_synthetic.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.output_parsers import StrOutputParser     # noqa: E402
from langchain_core.prompts import ChatPromptTemplate          # noqa: E402

from src.config import load_config                             # noqa: E402
from src.llm import get_llm                                    # noqa: E402
from src.vectorstore import get_vectorstore                    # noqa: E402

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write one exam-style question that this passage answers. The question must stand alone "
     "(no 'according to the passage'), and must NOT reuse distinctive phrases from the passage -- "
     "paraphrase instead, so it tests retrieval rather than string matching.\n"
     'Reply with JSON only: {{"question": "...", "type": "factual|conceptual"}}'),
    ("human", "Passage:\n{chunk}"),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "eval" / "questions_synthetic.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    raw = get_vectorstore(cfg).get(include=["documents", "metadatas"])
    pool = [(t, m) for t, m in zip(raw["documents"], raw["metadatas"]) if len(t) >= args.min_chars]
    if not pool:
        sys.exit("No chunks long enough -- build the index first.")
    random.Random(args.seed).shuffle(pool)

    llm, rows = get_llm(cfg), []
    for i, (text, meta) in enumerate(pool[: args.n], start=1):
        out = (PROMPT | llm | StrOutputParser()).invoke({"chunk": text[:2000]})
        out = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            d = json.loads(out)
        except Exception:
            continue
        rows.append({
            "id": f"s{i:03d}", "question": d["question"], "type": d.get("type", "factual"),
            "gold_doc": meta["doc"], "gold_pages": [meta["page"]], "split": "synthetic",
        })
        print(f"  s{i:03d} ({meta['doc']} p.{meta['page']}) {d['question'][:70]}")

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nwrote {len(rows)} synthetic questions -> {args.out}")


if __name__ == "__main__":
    main()
