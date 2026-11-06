"""
Bootstrap Consensus Trees (BCT; Eder, 2012, 2016) & Multi-MFW Stability Analysis.
Evaluates stylometric attribution across a grid of MFW thresholds and culling
ratios to ensure attributions are not artifacts of a single feature-set size.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np

from stylometry.delta import DeltaAnalyzer, DeltaMetric


class BootstrapConsensusAnalyzer:
    """
    Implements Maciej Eder's (2012, 2016) Bootstrap Consensus framework across
    multiple MFW sizes (e.g., M = 50, 100, 150, 200, 300, 400, 500) and culling
    thresholds.

    Produces:
    - Per-MFW attribution votes and consensus stability scores for disputed texts
    - Pairwise author-document consensus affinity matrix across the MFW grid
    """

    def __init__(
        self,
        mfw_grid: Sequence[int] = (50, 100, 150, 200, 250, 300, 400, 500),
        culling_grid: Sequence[float] = (0.0, 0.2),
        metrics: Sequence[DeltaMetric] = ("burrows", "cosine", "eder"),
    ):
        self.mfw_grid = list(mfw_grid)
        self.culling_grid = list(culling_grid)
        self.metrics = list(metrics)
        self.models_: List[Tuple[str, DeltaAnalyzer]] = []
        self.authors_: List[str] = []

    def fit(
        self,
        texts: Sequence[str],
        authors: Sequence[str],
    ) -> "BootstrapConsensusAnalyzer":
        self.authors_ = sorted(set(authors))
        self.models_ = []
        for mfw in self.mfw_grid:
            for cull in self.culling_grid:
                for metric in self.metrics:
                    mode = "document_centroid" if metric == "cosine" else "author_subcorpus"
                    analyzer = DeltaAnalyzer(
                        n_mfw=mfw,
                        metric=metric,
                        profile_mode=mode,
                        culling_ratio=cull,
                    )
                    analyzer.fit(texts, authors)
                    label = f"{metric}_mfw{mfw}_cull{int(cull*100)}"
                    self.models_.append((label, analyzer))
        return self

    def predict_consensus(self, text: str) -> Dict[str, object]:
        """
        Evaluate `text` across all `(MFW, culling, metric)` configurations in the
        bootstrap grid and return the consensus author and stability score.
        """
        votes: Dict[str, int] = {a: 0 for a in self.authors_}
        by_mfw: Dict[int, Dict[str, int]] = {m: {a: 0 for a in self.authors_} for m in self.mfw_grid}

        for label, model in self.models_:
            pred, _ = model.predict_one(text)
            votes[pred] = votes.get(pred, 0) + 1
            by_mfw[model.n_mfw][pred] = by_mfw[model.n_mfw].get(pred, 0) + 1

        total_runs = max(1, len(self.models_))
        proportions = {a: round(votes[a] / float(total_runs), 4) for a in self.authors_}
        best_author = max(votes, key=votes.get)

        return {
            "consensus_author": best_author,
            "stability_score": proportions[best_author],
            "total_bootstrap_models": total_runs,
            "vote_counts": votes,
            "vote_proportions": proportions,
            "votes_by_mfw": by_mfw,
        }
