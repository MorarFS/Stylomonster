#!/usr/bin/env python3
"""
Script 05: Attribute Any Custom Text or Disputed Document
---------------------------------------------------------
Standalone forensic stylometry script that trains all 12 attribution &
verification methods, Bootstrap Consensus Trees (48 models), Craig's Zeta,
and Rolling Stylometry on either:
  - A custom directory of reference `.txt` files (`--train-dir`), OR
  - One of the built-in benchmark corpora (`--corpus federalist` or `--corpus literature`)
and attributes any target query text (`--query`).

Example Usage:
  python scripts/05_attribute_custom_text.py --corpus federalist --paper 51
  python scripts/05_attribute_custom_text.py --corpus literature --query data/raw/novels/doyle_lost_world.txt
  python scripts/05_attribute_custom_text.py --train-dir path/to/known_authors --query path/to/disputed.txt
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from stylometry.classifiers import StylometricEnsembleAttributor
from stylometry.consensus_tree import BootstrapConsensusAnalyzer
from stylometry.corpus import (
    load_literary_novelists_corpus,
    parse_federalist_corpus,
    strip_gutenberg_boilerplate,
)
from stylometry.lexical import compute_lexical_metrics
from stylometry.rolling import RollingStylometryAnalyzer
from stylometry.syntactic import compute_syntactic_metrics


def load_training_data(args: argparse.Namespace, base_dir: Path) -> tuple[list[str], list[str], str, str]:
    if args.train_dir:
        ref_dir = Path(args.train_dir)
        texts, authors = [], []
        for p in sorted(ref_dir.rglob("*.txt")):
            author = p.parent.name if p.parent != ref_dir else p.stem.split("_")[0]
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            authors.append(author)
        q_path = Path(args.query)
        return texts, authors, q_path.read_text(encoding="utf-8", errors="ignore"), q_path.name

    if args.corpus == "federalist":
        docs = parse_federalist_corpus(base_dir / "data" / "raw" / "pg1404.txt")
        train_docs = [d for d in docs if d.status == "undisputed" and d.author in ("Hamilton", "Madison")]
        if args.paper:
            target = next(d for d in docs if d.metadata["paper_number"] == args.paper)
            q_text, q_label = target.text, target.title
        else:
            q_path = Path(args.query)
            q_text, q_label = q_path.read_text(encoding="utf-8", errors="ignore"), q_path.name
        return [d.text for d in train_docs], [d.author for d in train_docs], q_text, q_label

    # Literature corpus
    seg_docs = load_literary_novelists_corpus(
        base_dir / "data" / "raw" / "novels", segment_words=10000, max_segments_per_novel=3
    )
    train_docs = [d for d in seg_docs if d.status == "train"]
    if args.query:
        q_path = Path(args.query)
        q_text = strip_gutenberg_boilerplate(q_path.read_text(encoding="utf-8", errors="ignore"))
        q_label = q_path.name
    else:
        target = next(d for d in seg_docs if d.status == "test")
        q_text, q_label = target.text, target.title
    return [d.text for d in train_docs], [d.author for d in train_docs], q_text, q_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic stylometric attribution of a target document")
    parser.add_argument("--corpus", choices=["federalist", "literature"], default="federalist")
    parser.add_argument("--train-dir", help="Custom folder of reference author `.txt` files")
    parser.add_argument("--query", help="Path to unknown/disputed `.txt` file")
    parser.add_argument("--paper", type=int, default=51, help="Federalist paper number to test (default: 51)")
    parser.add_argument("--json-out", help="Optional path to save full JSON report")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    train_texts, train_authors, query_text, query_label = load_training_data(args, base_dir)

    ens = StylometricEnsembleAttributor(n_mfw=150, imposters_iterations=100)
    ens.fit(train_texts, train_authors)
    attr = ens.attribute_document(query_text)

    bct = BootstrapConsensusAnalyzer()
    bct.fit(train_texts, train_authors)
    bct_res = bct.predict_consensus(query_text)

    roller = RollingStylometryAnalyzer(n_mfw=100, window_size=500, step_size=250)
    roller.fit(train_texts, train_authors)
    roll_res = roller.analyze_text(query_text)

    lex = compute_lexical_metrics(query_text)
    syn = compute_syntactic_metrics(query_text)

    report = {
        "target_document": query_label,
        "candidate_authors": sorted(set(train_authors)),
        "word_count": int(lex["token_count"]),
        "consensus_author": attr["consensus_author"],
        "ensemble_agreement": attr["consensus_agreement"],
        "bootstrap_stability_48_models": bct_res["stability_score"],
        "method_predictions": attr["predictions"],
        "method_details": attr["details"],
        "rolling_window_proportions": roll_res["window_winner_proportions"],
        "lexical_summary": {
            k: round(lex[k], 4) for k in ("mean_word_length", "mattr", "yules_k", "honores_r")
        },
        "syntactic_summary": {
            k: round(syn[k], 4)
            for k in ("mean_sentence_length", "semicolon_rate_per_1k", "colon_rate_per_1k", "flesch_kincaid_grade")
        },
    }

    print("=" * 78)
    print(f"FORENSIC STYLOMETRY REPORT: {query_label} ({report['word_count']} words)")
    print("=" * 78)
    print(f"  Ensemble Consensus Author : {report['consensus_author']} ({report['ensemble_agreement']*100:.1f}% method vote)")
    print(f"  Bootstrap Stability (BCT) : {report['bootstrap_stability_48_models']*100:.1f}% across 48 MFW/Culling models")
    print(f"  Rolling Window Trajectory : {report['rolling_window_proportions']}")
    print("\n  Per-Method Predictions (12 Algorithms):")
    for m, pred in report["method_predictions"].items():
        print(f"    - {m:22s} -> {pred}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[OK] Saved JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
