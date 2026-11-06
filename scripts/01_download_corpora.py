#!/usr/bin/env python3
"""
Script 01: Download and Parse Benchmark Corpora
-----------------------------------------------
1. Downloads The Federalist Papers (`pg1404.txt`) and 15 classic 19th-century
   novels from Project Gutenberg if not already cached in `data/raw/`.
2. Parses all 85 Federalist Papers (stripping headers, salutations, and PUBLIUS
   signatures) and saves `data/processed/federalist_corpus.json`.
3. Parses and segments the 15 classic novels into 45 balanced 10,000-word
   literary segments and saves `data/processed/literature_segments_corpus.json`.
"""

from __future__ import annotations
import json
from pathlib import Path
import urllib.request

from stylometry.corpus import (
    NOVEL_METADATA,
    load_literary_novelists_corpus,
    parse_federalist_corpus,
)

GUTENBERG_NOVEL_IDS = {
    "austen_pride.txt": 1342,
    "austen_sense.txt": 161,
    "austen_emma.txt": 158,
    "bronte_jane_eyre.txt": 1260,
    "bronte_villette.txt": 9182,
    "bronte_shirley.txt": 30486,
    "dickens_great_expectations.txt": 1400,
    "dickens_two_cities.txt": 98,
    "dickens_oliver_twist.txt": 730,
    "doyle_sherlock.txt": 1661,
    "doyle_baskervilles.txt": 2852,
    "doyle_lost_world.txt": 139,
    "wells_time_machine.txt": 35,
    "wells_war_of_worlds.txt": 36,
    "wells_invisible_man.txt": 5230,
}


def download_if_missing(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "StylometryToolkit/1.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    novels_dir = raw_dir / "novels"
    proc_dir = base_dir / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    fed_raw = raw_dir / "pg1404.txt"
    download_if_missing("https://www.gutenberg.org/cache/epub/1404/pg1404.txt", fed_raw)

    for fname, gid in GUTENBERG_NOVEL_IDS.items():
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        download_if_missing(url, novels_dir / fname)

    fed_docs = parse_federalist_corpus(fed_raw)
    fed_out = proc_dir / "federalist_corpus.json"
    fed_out.write_text(
        json.dumps([d.to_dict() for d in fed_docs], indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Parsed {len(fed_docs)} Federalist Papers -> {fed_out}")

    lit_docs = load_literary_novelists_corpus(novels_dir, segment_words=10000, max_segments_per_novel=3)
    lit_out = proc_dir / "literature_segments_corpus.json"
    lit_out.write_text(
        json.dumps([d.to_dict() for d in lit_docs], indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Parsed {len(lit_docs)} literary novel segments -> {lit_out}")


if __name__ == "__main__":
    main()
