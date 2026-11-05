"""Download the corpus listed in corpus.yaml into data/raw/.

All sources are openly available typed (LaTeX-generated) PDFs from arXiv --
no handwritten notes, no scans, so no OCR path is needed. PDFs are gitignored;
anyone cloning the repo reruns this script.

    python scripts/fetch_corpus.py
"""
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"
UA = {"User-Agent": "notes-rag-kb/0.1 (student portfolio project)"}


def main():
    manifest = yaml.safe_load((ROOT / "corpus.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for p in manifest["papers"]:
        dest = OUT / p["file"]
        if dest.exists():
            print(f"skip  {p['file']} (exists)")
            continue
        url = f"https://arxiv.org/pdf/{p['id']}"
        print(f"fetch {p['file']:<34} <- {url}")
        try:
            r = requests.get(url, headers=UA, timeout=60)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                print(f"  !! not a PDF, skipping ({len(r.content)} bytes)")
                continue
            dest.write_bytes(r.content)
        except Exception as e:
            print(f"  !! failed: {e}")
        time.sleep(3)   # be polite to arXiv
    n = len(list(OUT.glob("*.pdf")))
    print(f"\n{n} PDFs in {OUT}")
    if n == 0:
        sys.exit("Nothing downloaded -- check your connection.")


if __name__ == "__main__":
    main()
