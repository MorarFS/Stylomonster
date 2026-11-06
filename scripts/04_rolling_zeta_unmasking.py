#!/usr/bin/env python3
"""
Script 04: Advanced Forensic Stylometry — Rolling Stylometry, Craig's Zeta,
Koppel's Unmasking, and Bootstrap Consensus Stability
---------------------------------------------------------------------------
Executes four specialized forensic stylometry analyses:
1. Burrows/Craig Zeta (ζ) Distinctive Vocabulary Analysis:
   - Extracts top mid-frequency lexical markers distinguishing Hamilton vs. Madison,
     Arthur Conan Doyle vs. H. G. Wells, and Charlotte Brontë vs. Jane Austen.
2. Eder's Rolling Stylometry (Sequential Window Analysis):
   - Tracks window-by-window attribution across the co-authored Federalist Nos. 18, 19, 20
     and disputed Federalist No. 51 and No. 55.
3. Koppel & Schler's 'Unmasking' Degradation Curves:
   - Compares iterative feature-elimination accuracy curves for:
     (a) Same-Author pair: Hamilton First Half vs. Hamilton Second Half
     (b) Different-Author pair: Hamilton vs. Madison
     (c) Disputed Essays vs. Hamilton (remains separable -> different authors!)
     (d) Disputed Essays vs. Madison (degrades rapidly -> same author!)
4. Eder's Bootstrap Consensus Tree (BCT) Multi-MFW Stability Analysis:
   - Evaluates all 12 disputed Federalist Papers across 48 parameter combinations
     (MFW in {50, 100, 150, 200, 250, 300, 400, 500} x Culling in {0%, 20%} x 3 Delta metrics).
"""

from __future__ import annotations
import json
from pathlib import Path

from stylometry.consensus_tree import BootstrapConsensusAnalyzer
from stylometry.corpus import load_literary_novelists_corpus, parse_federalist_corpus
from stylometry.rolling import RollingStylometryAnalyzer
from stylometry.zeta import UnmaskingVerifier, ZetaAnalyzer


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    proc_dir = base_dir / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    fed_docs = parse_federalist_corpus(raw_dir / "pg1404.txt")
    hm_docs = [d for d in fed_docs if d.status == "undisputed" and d.author in ("Hamilton", "Madison")]
    disputed_docs = [d for d in fed_docs if d.status == "disputed"]
    joint_docs = [d for d in fed_docs if d.status == "joint"]

    # 1. Craig's Zeta Distinctive Vocabulary (Hamilton vs. Madison)
    print("[1/4] Computing Burrows/Craig Zeta (ζ) distinctive vocabulary markers...")
    zeta_hm = ZetaAnalyzer(segment_size=1000, top_n_markers=30)
    zeta_hm.fit([d.text for d in hm_docs], [d.author for d in hm_docs])
    hm_zeta_markers = zeta_hm.get_contrastive_markers("Madison", top_k=15)

    # Also Zeta on Literature (Doyle vs. Wells sci-fi/adventure comparison)
    lit_docs = load_literary_novelists_corpus(raw_dir / "novels", segment_words=10000, max_segments_per_novel=3)
    dw_docs = [d for d in lit_docs if d.author in ("Arthur Conan Doyle", "H. G. Wells")]
    zeta_dw = ZetaAnalyzer(segment_size=1500, top_n_markers=30)
    zeta_dw.fit([d.text for d in dw_docs], [d.author for d in dw_docs])
    dw_zeta_markers = zeta_dw.get_contrastive_markers("Arthur Conan Doyle", top_k=15)

    # 2. Rolling Stylometry on Co-Authored (#18-20) and Disputed (#51, #55) Essays
    print("[2/4] Running Eder's Rolling Stylometry across co-authored & disputed papers...")
    roller = RollingStylometryAnalyzer(n_mfw=100, metric="burrows", window_size=500, step_size=150)
    roller.fit([d.text for d in hm_docs], [d.author for d in hm_docs])

    rolling_results = {}
    target_ids = {"fed_18", "fed_19", "fed_20", "fed_51", "fed_55"}
    for d in fed_docs:
        if d.doc_id in target_ids:
            rolling_results[d.doc_id] = roller.analyze_text(d.text)

    # 3. Koppel & Schler's Unmasking Verification Curves
    print("[3/4] Running Koppel & Schler's Unmasking degradation curves...")
    unmasker = UnmaskingVerifier(n_mfw=200, n_iterations=8, eliminate_per_side=4, chunk_size=800)
    h_texts = [d.text for d in hm_docs if d.author == "Hamilton"]
    m_texts = [d.text for d in hm_docs if d.author == "Madison"]
    disp_texts = [d.text for d in disputed_docs]

    unmasking_results = {
        "same_author_hamilton_split": unmasker.compute_degradation_curve(
            h_texts[:25], h_texts[25:]
        ),
        "different_author_hamilton_vs_madison": unmasker.compute_degradation_curve(
            h_texts[:18], m_texts
        ),
        "disputed_vs_hamilton": unmasker.compute_degradation_curve(
            disp_texts, h_texts[:18]
        ),
        "disputed_vs_madison": unmasker.compute_degradation_curve(
            disp_texts, m_texts
        ),
    }

    # 4. Bootstrap Consensus Tree (BCT) Stability across 48 parameter combinations
    print("[4/4] Running Bootstrap Consensus Tree (BCT) across 48 MFW/Culling/Metric models...")
    bct = BootstrapConsensusAnalyzer(
        mfw_grid=(50, 100, 150, 200, 250, 300, 400, 500),
        culling_grid=(0.0, 0.2),
        metrics=("burrows", "cosine", "eder"),
    )
    bct.fit([d.text for d in hm_docs], [d.author for d in hm_docs])

    bct_disputed = {}
    for d in disputed_docs:
        bct_disputed[d.doc_id] = bct.predict_consensus(d.text)

    payload = {
        "zeta_federalist_madison_vs_hamilton": hm_zeta_markers,
        "zeta_literature_doyle_vs_wells": dw_zeta_markers,
        "rolling_stylometry": rolling_results,
        "unmasking_verification": unmasking_results,
        "bootstrap_consensus_disputed_12": bct_disputed,
    }

    out_path = proc_dir / "advanced_forensics_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Saved advanced forensic stylometry results -> {out_path}")

    # Summary output
    print("\n--- TOP ZETA (ζ) DISTINCTIVE MARKERS: MADISON vs. HAMILTON ---")
    print("  Preferred by Madison :", [x["word"] + f" (ζ={x['zeta']:+.2f})" for x in hm_zeta_markers["preferred_by_Madison"][:10]])
    print("  Avoided by Madison (Hamilton markers):", [x["word"] + f" (ζ={x['zeta']:+.2f})" for x in hm_zeta_markers["avoided_by_Madison"][:10]])

    print("\n--- UNMASKING DEGRADATION CURVES (Initial -> Final Accuracy) ---")
    for name, res in unmasking_results.items():
        print(
            f"  {name:38s}: {res['initial_accuracy']*100:5.1f}% -> {res['final_accuracy']*100:5.1f}% "
            f"(Mean={res['mean_accuracy']*100:5.1f}%, MaxDrop={res['max_drop']*100:5.1f}%) | Curve={res['curve']}"
        )

    print("\n--- BOOTSTRAP CONSENSUS STABILITY (48 Models across MFW 50..500) ---")
    for doc_id, res in bct_disputed.items():
        print(
            f"  {doc_id}: Consensus={res['consensus_author']:8s} | "
            f"Stability={res['stability_score']*100:5.1f}% ({res['vote_counts']})"
        )


if __name__ == "__main__":
    main()
