#!/usr/bin/env python3
"""
Script 02: Extract Comprehensive Stylometric Feature Matrices
-------------------------------------------------------------
Extracts a tabular feature matrix for every document in:
1. The Federalist Papers (85 essays) -> `data/processed/federalist_features.csv`
2. 19th-Century Literature Corpus (45 segments) -> `data/processed/literature_features.csv`

Columns extracted per document:
- Metadata: `doc_id`, `title`, `author`, `status`, `word_count`
- 14 Lexical Richness Metrics (`ttr`, `root_ttr_guiraud`, `log_ttr_herdan`, `mattr`,
  `hapax_ratio`, `dislegomena_ratio`, `yules_k`, `simpsons_d`, `honores_r`, `brunets_w`, etc.)
- 22 Syntactic, Punctuation & Readability Metrics (`mean_sentence_length`,
  `semicolon_rate_per_1k`, `colon_rate_per_1k`, `dash_rate_per_1k`, `comma_rate_per_1k`,
  `flesch_kincaid_grade`, `gunning_fog_index`, etc.)
- 30 Mosteller & Wallace Function Word Marker Rates per 1,000 tokens (`mw_upon`,
  `mw_whilst`, `mw_while`, `mw_enough`, `mw_consequently`, etc.)
- 16 Mendenhall Word-Length Proportions (`mendenhall_len_01` .. `mendenhall_len_16`)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from stylometry.corpus import load_literary_novelists_corpus, parse_federalist_corpus
from stylometry.lexical import MendenhallCurveAnalyzer, compute_lexical_metrics
from stylometry.resources import MOSTELLER_WALLACE_MARKERS
from stylometry.syntactic import compute_function_word_rates, compute_syntactic_metrics


def build_feature_dataframe(docs) -> pd.DataFrame:
    mend = MendenhallCurveAnalyzer(max_word_length=16)
    rows = []
    for d in docs:
        lex = compute_lexical_metrics(d.text)
        syn = compute_syntactic_metrics(d.text)
        mw = compute_function_word_rates(d.text, word_list=MOSTELLER_WALLACE_MARKERS)
        curve = mend.compute_curve(d.text)

        row = {
            "doc_id": d.doc_id,
            "title": d.title,
            "author": d.author,
            "status": d.status,
            "word_count": d.word_count,
        }
        for k, v in lex.items():
            row[f"lex_{k}"] = round(float(v), 5)
        for k, v in syn.items():
            row[f"syn_{k}"] = round(float(v), 5)
        for w, v in mw.items():
            row[f"mw_{w}_per_1k"] = round(float(v), 5)
        for idx, val in enumerate(curve, start=1):
            row[f"mendenhall_len_{idx:02d}"] = round(float(val), 5)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    proc_dir = base_dir / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    fed_docs = parse_federalist_corpus(raw_dir / "pg1404.txt")
    df_fed = build_feature_dataframe(fed_docs)
    fed_csv = proc_dir / "federalist_features.csv"
    df_fed.to_csv(fed_csv, index=False)
    print(f"[OK] Extracted {df_fed.shape[0]} rows x {df_fed.shape[1]} features -> {fed_csv}")

    lit_docs = load_literary_novelists_corpus(raw_dir / "novels", segment_words=10000, max_segments_per_novel=3)
    df_lit = build_feature_dataframe(lit_docs)
    lit_csv = proc_dir / "literature_features.csv"
    df_lit.to_csv(lit_csv, index=False)
    print(f"[OK] Extracted {df_lit.shape[0]} rows x {df_lit.shape[1]} features -> {lit_csv}")


if __name__ == "__main__":
    main()
