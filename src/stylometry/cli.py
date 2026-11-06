"""
Command-line interface (`stylometry`) for inspecting stylometric resources,
profiling single texts, attributing unknown documents against a training folder,
running Rolling Stylometry, and executing corpus benchmarks.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from stylometry.classifiers import StylometricEnsembleAttributor
from stylometry.consensus_tree import BootstrapConsensusAnalyzer
from stylometry.lexical import MendenhallCurveAnalyzer, compute_lexical_metrics
from stylometry.resources import get_all_resources_summary, MOSTELLER_WALLACE_MARKERS
from stylometry.rolling import RollingStylometryAnalyzer
from stylometry.syntactic import compute_function_word_rates, compute_syntactic_metrics


def cmd_resources(args: argparse.Namespace) -> None:
    summary = get_all_resources_summary()
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("=" * 78)
    print("COMPUTATIONAL STYLOMETRY RESOURCES CATALOG")
    print("=" * 78)
    print(f"\n1. Implemented Stylometric Methods ({len(summary['methods'])}):")
    for m in summary["methods"]:
        print(f"   - [{m['year']}] {m['name']} ({m['authors']})")
        print(f"     Category: {m['category']} | Module: {m['implemented_in_module']}")
        print(f"     Formula : {m['formula_summary']}")

    print(f"\n2. External Open-Source Software & Frameworks ({len(summary['external_software'])}):")
    for s in summary["external_software"]:
        print(f"   - {s['name']} [{s['language']}] — {s['authors_or_org']}")
        print(f"     URL: {s['url']}")

    print(f"\n3. Benchmark Corpora ({len(summary['benchmark_corpora'])}):")
    for c in summary["benchmark_corpora"]:
        print(f"   - {c['name']} ({c['document_count']} texts): {', '.join(c['authors_included'])}")


def cmd_profile(args: argparse.Namespace) -> None:
    text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    lex = compute_lexical_metrics(text)
    syn = compute_syntactic_metrics(text)
    mw = compute_function_word_rates(text, word_list=MOSTELLER_WALLACE_MARKERS[:12])
    curve = MendenhallCurveAnalyzer().compute_curve(text).tolist()
    out = {
        "file": str(args.file),
        "lexical": {k: round(v, 4) for k, v in lex.items()},
        "syntactic": {k: round(v, 4) for k, v in syn.items()},
        "top_markers_per_1k": {k: round(v, 4) for k, v in mw.items()},
        "mendenhall_curve_1_to_16": [round(x, 4) for x in curve],
    }
    print(json.dumps(out, indent=2))


def _load_reference_dir(ref_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Load reference texts from a directory where files are named `<Author>_<Title>.txt`
    or organized in subdirectories `<ref_dir>/<Author>/<Title>.txt`.
    """
    texts, authors, doc_ids = [], [], []
    for p in sorted(ref_dir.rglob("*.txt")):
        if p.parent != ref_dir:
            author = p.parent.name
        else:
            author = p.stem.split("_")[0]
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        authors.append(author)
        doc_ids.append(p.stem)
    return texts, authors, doc_ids


def cmd_attribute(args: argparse.Namespace) -> None:
    ref_dir = Path(args.train_dir)
    query_path = Path(args.query)
    texts, authors, _ = _load_reference_dir(ref_dir)
    if len(set(authors)) < 2:
        raise SystemExit("Error: Reference directory must contain texts from at least 2 authors.")

    query_text = query_path.read_text(encoding="utf-8", errors="ignore")
    ens = StylometricEnsembleAttributor(n_mfw=args.mfw, imposters_iterations=100)
    ens.fit(texts, authors)
    res = ens.attribute_document(query_text)

    bct = BootstrapConsensusAnalyzer()
    bct.fit(texts, authors)
    bct_res = bct.predict_consensus(query_text)

    roll = RollingStylometryAnalyzer(n_mfw=min(100, args.mfw), window_size=500, step_size=250)
    roll.fit(texts, authors)
    roll_res = roll.analyze_text(query_text)

    payload = {
        "query_file": str(query_path),
        "candidate_authors": sorted(set(authors)),
        "ensemble_attribution": res,
        "bootstrap_consensus_stability": bct_res,
        "rolling_stylometry_summary": {
            "num_windows": roll_res["num_windows"],
            "window_winner_proportions": roll_res["window_winner_proportions"],
        },
    }
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stylometry",
        description="Unified computational stylometry and authorship attribution toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    p_res = sub.add_parser("resources", help="List all stylometric methods, word lists, and corpora")
    p_res.add_argument("--json", action="store_true", help="Output catalog as JSON")

    p_prof = sub.add_parser("profile", help="Extract lexical, syntactic, and Mendenhall profile from a text file")
    p_prof.add_argument("file", help="Path to text file")

    p_attr = sub.add_parser("attribute", help="Attribute an unknown text file against a directory of known author texts")
    p_attr.add_argument("--train-dir", required=True, help="Directory of reference `.txt` files (`Author_Title.txt` or `Author/Title.txt`)")
    p_attr.add_argument("--query", required=True, help="Path to disputed/unknown `.txt` file")
    p_attr.add_argument("--mfw", type=int, default=150, help="Number of Most Frequent Words (default: 150)")

    args = parser.parse_args(argv)
    if args.command == "resources":
        cmd_resources(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "attribute":
        cmd_attribute(args)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
