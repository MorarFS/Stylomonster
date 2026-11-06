"""
Rolling Stylometry / Sequential Stylometric Analysis (Eder, 2016) for detecting
co-authorship, collaborative passages, and stylistic shifts across a text.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Sequence
import numpy as np

from stylometry.delta import DeltaAnalyzer, DeltaMetric
from stylometry.preprocessing import chunk_tokens, tokenize_words


class RollingStylometryAnalyzer:
    """
    Implements Maciej Eder's (2016) Rolling Stylometry ('Sequential Stylometric
    Analysis').

    Given a reference corpus of candidate authors and a target text (such as a
    collaboratively written essay or novel):
    1. Standardizes the reference corpus on the top `n_mfw` words.
    2. Slides a window of `window_size` words across the target text in increments
       of `step_size` words.
    3. Computes the Delta distance (or relative attribution score) from each window
       to every candidate author, plotting a longitudinal stylistic trajectory
       across the text.
    """

    def __init__(
        self,
        n_mfw: int = 100,
        metric: DeltaMetric = "burrows",
        window_size: int = 500,
        step_size: int = 100,
    ):
        self.n_mfw = n_mfw
        self.metric = metric
        self.window_size = window_size
        self.step_size = step_size
        self.delta_ = DeltaAnalyzer(
            n_mfw=n_mfw, metric=metric, profile_mode="author_subcorpus"
        )

    def fit(self, texts: Sequence[str], authors: Sequence[str]) -> "RollingStylometryAnalyzer":
        self.delta_.fit(texts, authors)
        return self

    def analyze_text(self, text: str) -> Dict[str, object]:
        """
        Slide a window across `text` and return per-window distances and winner
        for each window position.
        """
        tokens = tokenize_words(text)
        n_tokens = len(tokens)
        if n_tokens == 0:
            return {"total_tokens": 0, "windows": [], "window_winner_counts": {}}

        if n_tokens <= self.window_size:
            windows = [(0, n_tokens, tokens)]
        else:
            windows = []
            for start in range(0, n_tokens - self.window_size + 1, self.step_size):
                end = start + self.window_size
                windows.append((start, end, tokens[start:end]))

        window_records: List[Dict[str, object]] = []
        winner_counts: Dict[str, int] = {}

        for idx, (start, end, win_tokens) in enumerate(windows):
            best_author, dists = self.delta_.predict_one(win_tokens)
            winner_counts[best_author] = winner_counts.get(best_author, 0) + 1
            window_records.append(
                {
                    "window_index": idx,
                    "start_token": start,
                    "end_token": end,
                    "center_token": (start + end) // 2,
                    "winner": best_author,
                    "distances": {k: round(float(v), 4) for k, v in dists.items()},
                }
            )

        total_wins = max(1, len(window_records))
        winner_proportions = {
            k: round(v / float(total_wins), 4) for k, v in winner_counts.items()
        }

        return {
            "total_tokens": n_tokens,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "num_windows": len(window_records),
            "window_winner_counts": winner_counts,
            "window_winner_proportions": winner_proportions,
            "windows": window_records,
        }
