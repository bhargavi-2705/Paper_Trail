"""Run the eval under several configs and print a comparison table.

    python scripts/ablations.py                 # retrieval-only ablations (free)
    python scripts/ablations.py --generation    # includes agent vs baseline (uses API)

Writes eval/results/<name>.json per variant and eval/results/ablations.md.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

from src.config import load_config  # noqa: E402

RETRIEVAL_VARIANTS = {
    "dense_only":       {"retrieval": {"hybrid": False, "rerank": False}},
    "hybrid":           {"retrieval": {"hybrid": True,  "rerank": False}},
    "dense_rerank":     {"retrieval": {"hybrid": False, "rerank": True}},
    "hybrid_rerank":    {"retrieval": {"hybrid": True,  "rerank": True}},
    "chunk_600":        {"retrieval": {"hybrid": True, "rerank": True},
                         "chunking": {"chunk_chars": 600, "overlap_chars": 100}},
    "chunk_2000":       {"retrieval": {"hybrid": True, "rerank": True},
                         "chunking": {"chunk_chars": 2000, "overlap_chars": 300}},
    "per_page":         {"retrieval": {"hybrid": True, "rerank": True},
                         "chunking": {"strategy": "per_page"}},
}
GENERATION_VARIANTS = {
    "baseline_chain":   {"agent": {"enabled": False}},
    "agent_no_critic":  {"agent": {"enabled": True, "decompose": True, "critic": False}},
    "agent_full":       {"agent": {"enabled": True, "decompose": True, "critic": True}},
}
COLS = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr", "faithfulness_mean",
        "retrieval_latency_s_mean", "answer_latency_s_mean"]


def deep_update(base, patch):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def run_variant(name, patch, generation, reindex):
    cfg = deep_update(copy.deepcopy(load_config()), patch)
    tmp = ROOT / ".ablation_config.yaml"
    tmp.write_text(yaml.safe_dump(cfg))
    env_cmd = [sys.executable]
    if reindex:
        print(f"[{name}] rebuilding index (chunking changed)")
        subprocess.run(env_cmd + ["-c",
            f"import sys; sys.path.insert(0,r'{ROOT}');"
            "from src.config import load_config; from src.build_index import build_index;"
            f"build_index(load_config(r'{tmp}'))"], check=True)
    cmd = env_cmd + [str(ROOT / "scripts" / "run_eval.py"), "--tag", name, "--config", str(tmp)]
    if generation:
        cmd.append("--generation")
    subprocess.run(cmd, check=True)
    return json.loads((ROOT / "eval" / "results" / f"{name}.json").read_text())["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    variants = dict(RETRIEVAL_VARIANTS)
    if args.generation:
        variants.update(GENERATION_VARIANTS)
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}

    rows = {}
    for name, patch in variants.items():
        reindex = "chunking" in patch
        try:
            rows[name] = run_variant(name, patch, args.generation, reindex)
        except subprocess.CalledProcessError as e:
            print(f"[{name}] FAILED: {e}")

    lines = ["| variant | " + " | ".join(COLS) + " |",
             "|---" * (len(COLS) + 1) + "|"]
    for name, s in rows.items():
        cells = []
        for c in COLS:
            v = s.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (ROOT / "eval" / "results" / "ablations.md").write_text(table + "\n")
    print("\n" + table)
    print("\nPaste this into docs/DECISIONS.md with a sentence on what it means.")


if __name__ == "__main__":
    main()
