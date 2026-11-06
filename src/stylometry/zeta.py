"""
Craig's / Burrows's Zeta (ζ) Distinctive Vocabulary Analysis (Burrows, 2007;
Craig & Kinney, 2009) and Koppel & Schler's (2004) 'Unmasking' Authorship
Verification via Iterative Feature Elimination.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from stylometry.preprocessing import chunk_tokens, select_mfw, tokenize_words


class ZetaAnalyzer:
    """
    Implements Burrows's (2007) and Craig & Kinney's (2009) Zeta (ζ) method for
    identifying distinctive marker words that consistently appear in one author's
    segments while being avoided by another author.

    For two groups of text segments S_A (Author A) and S_B (Author B):
      - d_A(w) = fraction of segments in S_A containing word w at least once
      - d_B(w) = fraction of segments in S_B containing word w at least once
      - Craig's Zeta: ζ(w) = d_A(w) - d_B(w) in [-1.0, +1.0]
      - Log2 Zeta Ratio: log2( (d_A(w) + α) / (d_B(w) + α) )

    Words with ζ(w) near +1.0 are signature markers of Author A; words with
    ζ(w) near -1.0 are signature markers of Author B.
    """

    def __init__(
        self,
        segment_size: int = 1000,
        top_n_markers: int = 50,
        min_doc_proportion: float = 0.15,
    ):
        self.segment_size = segment_size
        self.top_n_markers = top_n_markers
        self.min_doc_proportion = min_doc_proportion

        self.zeta_scores_: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.positive_markers_: Dict[str, List[Tuple[str, float]]] = {}
        self.negative_markers_: Dict[str, List[Tuple[str, float]]] = {}
        self.author_centroids_: Dict[str, Tuple[float, float]] = {}
        self.authors_: List[str] = []

    def _segment_author_texts(self, texts: Sequence[str]) -> List[Set[str]]:
        segments: List[Set[str]] = []
        for t in texts:
            toks = tokenize_words(t)
            if len(toks) <= self.segment_size:
                if toks:
                    segments.append(set(toks))
            else:
                chunks = chunk_tokens(
                    toks,
                    chunk_size=self.segment_size,
                    step_size=self.segment_size,
                    drop_Incomplete=False,
                )
                for c in chunks:
                    if len(c) >= max(100, self.segment_size // 3):
                        segments.append(set(c))
        return segments

    def fit(self, texts: Sequence[str], authors: Sequence[str]) -> "ZetaAnalyzer":
        unique_authors = sorted(set(authors))
        self.authors_ = unique_authors

        author_segments: Dict[str, List[Set[str]]] = {}
        vocab: Counter[str] = Counter()
        for a in unique_authors:
            a_texts = [t for t, auth in zip(texts, authors) if auth == a]
            segs = self._segment_author_texts(a_texts)
            author_segments[a] = segs
            for s in segs:
                vocab.update(s)

        total_segs = sum(len(s) for s in author_segments.values())
        candidate_words = [
            w for w, c in vocab.items() if c >= max(2, int(self.min_doc_proportion * total_segs))
        ]

        # One-vs-Rest Zeta for each author
        for a in unique_authors:
            segs_a = author_segments[a]
            segs_rest = [s for other_a, s_list in author_segments.items() if other_a != a for s in s_list]
            n_a = max(1, len(segs_a))
            n_r = max(1, len(segs_rest))

            scores: Dict[str, float] = {}
            for w in candidate_words:
                prop_a = sum(1 for s in segs_a if w in s) / float(n_a)
                prop_r = sum(1 for s in segs_rest if w in s) / float(n_r)
                scores[w] = prop_a - prop_r

            sorted_pos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            sorted_neg = sorted(scores.items(), key=lambda x: x[1])

            self.positive_markers_[a] = sorted_pos[: self.top_n_markers]
            self.negative_markers_[a] = sorted_neg[: self.top_n_markers]
            self.zeta_scores_[(a, "REST")] = scores

        # Compute training document coordinates in (pos_marker_coverage, neg_marker_coverage) space
        for a in unique_authors:
            pos_set = {w for w, _ in self.positive_markers_[a]}
            neg_set = {w for w, _ in self.negative_markers_[a]}
            a_texts = [t for t, auth in zip(texts, authors) if auth == a]
            coords = [self._score_doc_for_author(t, pos_set, neg_set) for t in a_texts]
            self.author_centroids_[a] = (
                float(np.mean([c[0] for c in coords])),
                float(np.mean([c[1] for c in coords])),
            )

        return self

    @staticmethod
    def _score_doc_for_author(
        text: str, pos_set: Set[str], neg_set: Set[str]
    ) -> Tuple[float, float]:
        types = set(tokenize_words(text))
        if not types:
            return (0.0, 0.0)
        pos_hit = len(types & pos_set) / float(max(1, len(pos_set)))
        neg_hit = len(types & neg_set) / float(max(1, len(neg_set)))
        return (pos_hit, neg_hit)

    def predict_one(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Compute net Zeta preference score for each candidate author:
        score(A) = proportion of A's positive Zeta markers present minus
                   proportion of A's negative (anti-A) Zeta markers present.
        """
        scores: Dict[str, float] = {}
        for a in self.authors_:
            pos_set = {w for w, _ in self.positive_markers_[a]}
            neg_set = {w for w, _ in self.negative_markers_[a]}
            pos_hit, neg_hit = self._score_doc_for_author(text, pos_set, neg_set)
            scores[a] = float(pos_hit - neg_hit)
        best = max(scores, key=scores.get)
        return best, scores

    def predict(self, texts: Sequence[str]) -> List[str]:
        return [self.predict_one(t)[0] for t in texts]

    def get_contrastive_markers(
        self, author_a: str, top_k: int = 15
    ) -> Dict[str, List[Dict[str, object]]]:
        """Return the top positive (preferred) and negative (avoided) Zeta words for `author_a`."""
        return {
            "preferred_by_" + author_a: [
                {"word": w, "zeta": round(float(z), 4)}
                for w, z in self.positive_markers_[author_a][:top_k]
            ],
            "avoided_by_" + author_a: [
                {"word": w, "zeta": round(float(z), 4)}
                for w, z in self.negative_markers_[author_a][:top_k]
            ],
        }


class UnmaskingVerifier:
    """
    Implements Koppel & Schler's (2004) 'Unmasking' method for authorship verification.

    Core insight ('Depth of Difference'):
    If two sets of text chunks A and B were written by the SAME author, a linear
    classifier can initially separate them with high cross-validation accuracy
    using a handful of topical/thematic words. However, when the top `k` most
    strongly weighted positive and negative features are iteratively removed at
    each round, accuracy drops steeply.
    Conversely, if A and B were written by DIFFERENT authors, deep stylistic
    differences allow the classifier to maintain high accuracy across many
    feature-elimination iterations.
    """

    def __init__(
        self,
        n_mfw: int = 250,
        n_iterations: int = 8,
        eliminate_per_side: int = 3,
        chunk_size: int = 750,
    ):
        self.n_mfw = n_mfw
        self.n_iterations = n_iterations
        self.eliminate_per_side = eliminate_per_side
        self.chunk_size = chunk_size

    def compute_degradation_curve(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
    ) -> Dict[str, object]:
        """
        Compute the Unmasking cross-validation accuracy degradation curve between
        text collection A and text collection B across `n_iterations` rounds of
        top-feature elimination.
        """
        tokens_a = [w for t in texts_a for w in tokenize_words(t)]
        tokens_b = [w for t in texts_b for w in tokenize_words(t)]

        chunks_a = chunk_tokens(tokens_a, chunk_size=self.chunk_size, drop_Incomplete=False)
        chunks_b = chunk_tokens(tokens_b, chunk_size=self.chunk_size, drop_Incomplete=False)

        chunks_a = [c for c in chunks_a if len(c) >= self.chunk_size // 2]
        chunks_b = [c for c in chunks_b if len(c) >= self.chunk_size // 2]

        # Balance chunk counts if heavily skewed
        max_chunks = min(max(len(chunks_a), len(chunks_b)), 30)
        chunks_a = chunks_a[:max_chunks]
        chunks_b = chunks_b[:max_chunks]

        all_chunks = chunks_a + chunks_b
        y = np.array([0] * len(chunks_a) + [1] * len(chunks_b), dtype=int)

        vocab = select_mfw(all_chunks, n_mfw=self.n_mfw)
        active_indices = list(range(len(vocab)))

        X_full = np.zeros((len(all_chunks), len(vocab)), dtype=float)
        for i, ch in enumerate(all_chunks):
            c = Counter(ch)
            n = float(len(ch))
            for j, w in enumerate(vocab):
                X_full[i, j] = c.get(w, 0) / n

        curve: List[float] = []
        eliminated_words_per_round: List[List[str]] = []

        n_splits = min(5, len(chunks_a), len(chunks_b))
        if n_splits < 2:
            return {"curve": [], "mean_accuracy": 0.0, "max_drop": 0.0, "eliminated": []}

        for _ in range(self.n_iterations):
            if len(active_indices) < 2 * self.eliminate_per_side + 2:
                break

            X_sub = X_full[:, active_indices]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sub)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_accs = []
            for tr, te in skf.split(X_scaled, y):
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                clf.fit(X_scaled[tr], y[tr])
                fold_accs.append(float(np.mean(clf.predict(X_scaled[te]) == y[te])))

            curve.append(round(float(np.mean(fold_accs)), 4))

            # Fit on full active set to find strongest positive & negative weights
            clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf_full.fit(X_scaled, y)
            weights = clf_full.coef_[0]

            sorted_pos_local = np.argsort(weights)[::-1][: self.eliminate_per_side]
            sorted_neg_local = np.argsort(weights)[: self.eliminate_per_side]
            to_remove_local = set(sorted_pos_local.tolist() + sorted_neg_local.tolist())

            removed_words = [vocab[active_indices[idx]] for idx in to_remove_local]
            eliminated_words_per_round.append(removed_words)

            active_indices = [
                global_idx
                for local_idx, global_idx in enumerate(active_indices)
                if local_idx not in to_remove_local
            ]

        max_drop = float(curve[0] - min(curve)) if curve else 0.0
        slope = float((curve[-1] - curve[0]) / max(1, len(curve) - 1)) if len(curve) > 1 else 0.0

        return {
            "curve": curve,
            "initial_accuracy": curve[0] if curve else 0.0,
            "final_accuracy": curve[-1] if curve else 0.0,
            "mean_accuracy": round(float(np.mean(curve)), 4) if curve else 0.0,
            "max_drop": round(max_drop, 4),
            "degradation_slope": round(slope, 4),
            "eliminated_words_by_round": eliminated_words_per_round,
        }
