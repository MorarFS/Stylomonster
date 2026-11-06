"""
End-to-end experimental evaluation of the `stylometry` package on two benchmark corpora:
1. The Federalist Papers (85 essays: Hamilton, Madison, Jay, 12 Disputed, 3 Joint)
2. 19th-Century Classic Literature Corpus (15 novels / 45 segments across Austen,
   Charlotte Brontë, Dickens, Arthur Conan Doyle, and H. G. Wells)
Saves full structured results to `data/processed/benchmark_results.json`.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np

from stylometry import (
    DeltaAnalyzer,
    MendenhallCurveAnalyzer,
    MOSTELLER_WALLACE_MARKERS,
    StylometricEnsembleAttributor,
    compute_function_word_rates,
    compute_lexical_metrics,
    compute_pca_and_clustering,
    compute_syntactic_metrics,
    evaluate_methods_cross_validation,
    get_all_resources_summary,
    load_literary_novelists_corpus,
    parse_federalist_corpus,
)


def summarize_author_profiles(texts: list[str], authors: list[str], key_markers: list[str]) -> dict:
    """Compute mean lexical, syntactic, Mendenhall, and marker-word profiles per author."""
    unique_authors = sorted(set(authors))
    mend = MendenhallCurveAnalyzer(max_word_length=15)
    mend.fit(texts, authors)

    profiles = {}
    for a in unique_authors:
        a_texts = [t for t, auth in zip(texts, authors) if auth == a]
        lex_list = [compute_lexical_metrics(t) for t in a_texts]
        syn_list = [compute_syntactic_metrics(t) for t in a_texts]
        fw_list = [compute_function_word_rates(t, word_list=key_markers) for t in a_texts]

        lex_keys = list(lex_list[0].keys())
        syn_keys = list(syn_list[0].keys())

        profiles[a] = {
            "document_count": len(a_texts),
            "total_words": int(sum(l["token_count"] for l in lex_list)),
            "lexical_means": {
                k: round(float(np.mean([l[k] for l in lex_list])), 4) for k in lex_keys
            },
            "syntactic_means": {
                k: round(float(np.mean([s[k] for s in syn_list])), 4) for k in syn_keys
            },
            "marker_rates_per_1k": {
                w: round(float(np.mean([f[w] for f in fw_list])), 4) for w in key_markers
            },
            "mendenhall_curve_1_to_15": [
                round(float(x), 4) for x in mend.author_curves_[a]
            ],
        }
    return profiles


def run_federalist_experiments(base_dir: Path) -> dict:
    raw_path = base_dir / "data" / "raw" / "pg1404.txt"
    docs = parse_federalist_corpus(raw_path)

    undisputed_all = [d for d in docs if d.status == "undisputed"]
    undisputed_hm = [d for d in undisputed_all if d.author in ("Hamilton", "Madison")]
    disputed_docs = [d for d in docs if d.status == "disputed"]
    joint_docs = [d for d in docs if d.status == "joint"]

    key_markers = [
        "upon", "on", "while", "whilst", "enough", "there", "consequently",
        "commonly", "by", "to", "from", "in", "also", "an", "both", "though",
        "considerable", "according", "vigor", "innovation", "direction", "apt",
    ]

    # 1A. Author Stylometric Profiles (Hamilton, Madison, Jay)
    author_profiles = summarize_author_profiles(
        [d.text for d in undisputed_all],
        [d.author for d in undisputed_all],
        key_markers=key_markers,
    )

    # 1B. Cross-Validation on 70 Undisputed (3-class: Hamilton, Madison, Jay)
    cv_3class = evaluate_methods_cross_validation(
        [d.text for d in undisputed_all],
        [d.author for d in undisputed_all],
        n_splits=5,
        n_mfw=150,
    )

    # Cross-Validation on 65 Undisputed (2-class: Hamilton vs. Madison)
    cv_2class = evaluate_methods_cross_validation(
        [d.text for d in undisputed_hm],
        [d.author for d in undisputed_hm],
        n_splits=5,
        n_mfw=150,
    )

    # 1C. Sanity check: Federalist No. 64 (Historically verified John Jay)
    train_no64 = [d for d in undisputed_all if d.doc_id != "fed_64"]
    doc_64 = next(d for d in undisputed_all if d.doc_id == "fed_64")
    ens_3class = StylometricEnsembleAttributor(n_mfw=150, imposters_iterations=100)
    ens_3class.fit([d.text for d in train_no64], [d.author for d in train_no64])
    fed64_attribution = ens_3class.attribute_document(doc_64.text)

    # 1D. Resolve the 12 Disputed Federalist Papers (#49-58, #62-63) & 3 Joint Papers (#18-20)
    ens_hm = StylometricEnsembleAttributor(n_mfw=150, imposters_iterations=100)
    ens_hm.fit([d.text for d in undisputed_hm], [d.author for d in undisputed_hm])

    disputed_results = []
    for d in disputed_docs:
        attr = ens_hm.attribute_document(d.text)
        top_features = ens_hm.delta_burrows.explain_attribution(
            d.text, "Madison", "Hamilton", top_n=6
        )
        fw_rates = compute_function_word_rates(
            d.text, word_list=["upon", "on", "while", "whilst", "enough", "consequently"]
        )
        disputed_results.append(
            {
                "doc_id": d.doc_id,
                "paper_number": d.metadata["paper_number"],
                "title": d.title,
                "word_count": d.word_count,
                "consensus_author": attr["consensus_author"],
                "consensus_agreement": attr["consensus_agreement"],
                "vote_breakdown": attr["vote_breakdown"],
                "predictions": attr["predictions"],
                "burrows_delta": attr["details"]["burrows_delta_distances"],
                "cosine_delta": attr["details"]["cosine_delta_distances"],
                "quadratic_delta": attr["details"]["quadratic_delta_distances"],
                "eder_delta": attr["details"]["eder_delta_distances"],
                "kilgarriff_chi2": attr["details"]["kilgarriff_chi2_distances"],
                "mosteller_wallace_probs": attr["details"]["mosteller_wallace_probs"],
                "general_imposters_scores": attr["details"]["general_imposters_scores"],
                "composite_ml_probs": attr["details"]["composite_ml_probs"],
                "marker_rates_per_1k": {k: round(v, 3) for k, v in fw_rates.items()},
                "top_discriminating_mfw": top_features,
            }
        )

    joint_results = []
    for d in joint_docs:
        attr = ens_hm.attribute_document(d.text)
        joint_results.append(
            {
                "doc_id": d.doc_id,
                "paper_number": d.metadata["paper_number"],
                "title": d.title,
                "word_count": d.word_count,
                "consensus_author": attr["consensus_author"],
                "consensus_agreement": attr["consensus_agreement"],
                "predictions": attr["predictions"],
                "burrows_delta": attr["details"]["burrows_delta_distances"],
                "cosine_delta": attr["details"]["cosine_delta_distances"],
                "mosteller_wallace_probs": attr["details"]["mosteller_wallace_probs"],
                "general_imposters_scores": attr["details"]["general_imposters_scores"],
            }
        )

    # 1E. PCA & Hierarchical Clustering over all 85 Federalist Papers
    pca_labels = []
    for d in docs:
        if d.status == "disputed":
            pca_labels.append("Disputed")
        elif d.status == "joint":
            pca_labels.append("Joint")
        else:
            pca_labels.append(d.author)

    pca_res = compute_pca_and_clustering(
        [d.text for d in docs],
        pca_labels,
        [d.doc_id for d in docs],
        n_mfw=150,
    )

    return {
        "corpus_summary": {
            "total_papers": len(docs),
            "undisputed_hamilton": 51,
            "undisputed_madison": 14,
            "undisputed_jay": 5,
            "disputed_papers": 12,
            "joint_papers": 3,
            "total_words": int(sum(d.word_count for d in docs)),
        },
        "author_profiles": author_profiles,
        "cross_validation_3class_70_papers": cv_3class,
        "cross_validation_2class_65_papers": cv_2class,
        "federalist_64_jay_test": fed64_attribution,
        "disputed_12_papers_attribution": disputed_results,
        "joint_3_papers_attribution": joint_results,
        "pca_analysis": {
            "explained_variance_ratio": pca_res["explained_variance_ratio"],
            "pc1_top_loadings": pca_res["pc1_top_loadings"][:10],
            "pc2_top_loadings": pca_res["pc2_top_loadings"][:10],
            "points": pca_res["points"],
        },
    }


def run_literature_experiments(base_dir: Path) -> dict:
    novels_dir = base_dir / "data" / "raw" / "novels"
    seg_docs = load_literary_novelists_corpus(
        novels_dir, segment_words=10000, max_segments_per_novel=3
    )

    key_markers = [
        "upon", "on", "while", "whilst", "very", "quite", "rather", "indeed",
        "perhaps", "never", "always", "already", "towards", "toward", "within",
        "without", "though", "although",
    ]

    # 2A. Author Fingerprints across 45 segments (5 novelists)
    author_profiles = summarize_author_profiles(
        [d.text for d in seg_docs],
        [d.author for d in seg_docs],
        key_markers=key_markers,
    )

    # 2B. 5-Fold Stratified Cross-Validation across all 45 10,000-word segments
    cv_5authors = evaluate_methods_cross_validation(
        [d.text for d in seg_docs],
        [d.author for d in seg_docs],
        n_splits=5,
        n_mfw=150,
    )

    # 2C. Held-out Unseen Novel Attribution:
    # Train on 2 novels (6 segments) per author = 30 segments,
    # Test on 1 completely unseen novel (3 segments + full novel) per author!
    train_docs = [d for d in seg_docs if d.status == "train"]
    test_docs = [d for d in seg_docs if d.status == "test"]

    ens_lit = StylometricEnsembleAttributor(n_mfw=150, imposters_iterations=100)
    ens_lit.fit([d.text for d in train_docs], [d.author for d in train_docs])

    heldout_segment_results = []
    for d in test_docs:
        attr = ens_lit.attribute_document(d.text)
        heldout_segment_results.append(
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "novel": d.metadata["novel"],
                "true_author": d.author,
                "consensus_author": attr["consensus_author"],
                "correct": attr["consensus_author"] == d.author,
                "consensus_agreement": attr["consensus_agreement"],
                "predictions": attr["predictions"],
                "burrows_delta": attr["details"]["burrows_delta_distances"],
                "cosine_delta": attr["details"]["cosine_delta_distances"],
                "general_imposters_scores": attr["details"]["general_imposters_scores"],
            }
        )

    # Also evaluate full held-out novels (pooling the 3 segments of each held-out novel)
    full_novels = load_literary_novelists_corpus(novels_dir, segment_words=None)
    heldout_full_novels = [d for d in full_novels if d.status == "test"]
    heldout_novel_summary = []
    for d in heldout_full_novels:
        attr = ens_lit.attribute_document(d.text)
        heldout_novel_summary.append(
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "true_author": d.author,
                "word_count": d.word_count,
                "consensus_author": attr["consensus_author"],
                "correct": attr["consensus_author"] == d.author,
                "consensus_agreement": attr["consensus_agreement"],
                "predictions": attr["predictions"],
                "burrows_delta": attr["details"]["burrows_delta_distances"],
                "cosine_delta": attr["details"]["cosine_delta_distances"],
                "kilgarriff_chi2": attr["details"]["kilgarriff_chi2_distances"],
                "general_imposters_scores": attr["details"]["general_imposters_scores"],
            }
        )

    pca_lit = compute_pca_and_clustering(
        [d.text for d in seg_docs],
        [d.author for d in seg_docs],
        [d.doc_id for d in seg_docs],
        n_mfw=150,
    )

    return {
        "corpus_summary": {
            "authors": sorted(set(d.author for d in seg_docs)),
            "total_novels": 15,
            "total_segments": len(seg_docs),
            "words_per_segment": 10000,
        },
        "author_profiles": author_profiles,
        "cross_validation_45_segments": cv_5authors,
        "heldout_novels_attribution": heldout_novel_summary,
        "heldout_segments_attribution": heldout_segment_results,
        "pca_analysis": {
            "explained_variance_ratio": pca_lit["explained_variance_ratio"],
            "pc1_top_loadings": pca_lit["pc1_top_loadings"][:10],
            "pc2_top_loadings": pca_lit["pc2_top_loadings"][:10],
            "points": pca_lit["points"],
        },
    }


def main() -> None:
    t0 = time.time()
    base_dir = Path("/home/antigravity/stylometry_toolkit")
    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Running The Federalist Papers stylometric experiments...")
    fed_results = run_federalist_experiments(base_dir)

    print("[2/2] Running 19th-Century Classic Literature stylometric experiments...")
    lit_results = run_literature_experiments(base_dir)

    full_payload = {
        "generated_at_seconds": round(time.time() - t0, 2),
        "resources_catalog": get_all_resources_summary(),
        "federalist_benchmark": fed_results,
        "literature_benchmark": lit_results,
    }

    out_file = out_dir / "benchmark_results.json"
    out_file.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    print(f"Done in {full_payload['generated_at_seconds']}s! Saved to {out_file}")

    # Print key summary to stdout
    print("\n=== FEDERALIST 2-CLASS CV (Hamilton vs Madison, 65 Undisputed Papers) ===")
    for m, stats in fed_results["cross_validation_2class_65_papers"].items():
        print(f"  {m:22s}: Accuracy={stats['accuracy']*100:6.2f}% ({stats['correct']}/{stats['total']}) | Macro-F1={stats['macro_f1']:.4f}")

    print("\n=== FEDERALIST 3-CLASS CV (Hamilton, Madison, Jay, 70 Undisputed Papers) ===")
    for m, stats in fed_results["cross_validation_3class_70_papers"].items():
        print(f"  {m:22s}: Accuracy={stats['accuracy']*100:6.2f}% ({stats['correct']}/{stats['total']}) | Macro-F1={stats['macro_f1']:.4f}")

    print("\n=== FEDERALIST NO. 64 SANITY CHECK (True Author: Jay) ===")
    print("  Consensus:", fed_results["federalist_64_jay_test"]["consensus_author"],
          "| Agreement:", fed_results["federalist_64_jay_test"]["consensus_agreement"])
    print("  Predictions:", fed_results["federalist_64_jay_test"]["predictions"])

    print("\n=== 12 DISPUTED FEDERALIST PAPERS (#49-58, #62-63) ===")
    for item in fed_results["disputed_12_papers_attribution"]:
        print(
            f"  Paper #{item['paper_number']:2d} ({item['word_count']:4d}w) -> "
            f"Consensus: {item['consensus_author']:8s} ({item['consensus_agreement']*100:5.1f}% vote) | "
            f"Burrows(H={item['burrows_delta']['Hamilton']:.3f}, M={item['burrows_delta']['Madison']:.3f}) | "
            f"Cosine(H={item['cosine_delta']['Hamilton']:.3f}, M={item['cosine_delta']['Madison']:.3f}) | "
            f"MW_P(M)={item['mosteller_wallace_probs']['Madison']:.3f} | "
            f"GI(M)={item['general_imposters_scores']['Madison']:.2f}"
        )

    print("\n=== LITERATURE 5-AUTHOR CV (45 x 10k-word segments) ===")
    for m, stats in lit_results["cross_validation_45_segments"].items():
        print(f"  {m:22s}: Accuracy={stats['accuracy']*100:6.2f}% ({stats['correct']}/{stats['total']}) | Macro-F1={stats['macro_f1']:.4f}")

    print("\n=== HELD-OUT UNSEEN NOVELS ATTRIBUTION (5 Novels) ===")
    for item in lit_results["heldout_novels_attribution"]:
        print(
            f"  {item['title']:28s} (True: {item['true_author']:18s}) -> "
            f"Predicted: {item['consensus_author']:18s} "
            f"[Agreement: {item['consensus_agreement']*100:5.1f}%]"
        )


if __name__ == "__main__":
    main()
