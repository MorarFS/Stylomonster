"""
Koppel & Winter's (2014) General Imposters (GI) Method for Open-Set
Authorship Verification (as also popularized in Kestemont et al., 2016 and R's `stylo::imposters`).
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np

from stylometry.preprocessing import select_mfw, tokenize_words


class GeneralImpostersVerifier:
    """
    Implements the General Imposters (GI) authorship verification framework
    (Koppel & Winter, 2014; Kestemont et al., 2016).

    Given a disputed document X, a candidate author C, and a reference set of
    distractor ('imposter') documents I from other authors:
    In each of `n_iterations` Monte Carlo trials:
      1. Randomly sample a subset of `feature_sample_ratio` (default 50%) of the
         M Most Frequent Words (or character n-grams).
      2. Randomly sample a subset of imposter documents.
      3. Measure whether X is stylistically closer (via Cosine or MinMax distance
         on the sampled feature subset) to at least one reference document of
         candidate C than to the closest sampled imposter document.
    The verification score S(X, C) in [0.0, 1.0] is the proportion of trials
    in which candidate C wins. Typically S > 0.50 confirms authorship.
    """

    def __init__(
        self,
        n_mfw: int = 250,
        n_iterations: int = 100,
        feature_sample_ratio: float = 0.5,
        imposter_sample_ratio: float = 0.7,
        distance: Literal["cosine", "minmax", "manhattan"] = "cosine",
        random_state: int = 42,
    ):
        self.n_mfw = n_mfw
        self.n_iterations = n_iterations
        self.feature_sample_ratio = feature_sample_ratio
        self.imposter_sample_ratio = imposter_sample_ratio
        self.distance = distance
        self.random_state = random_state

        self.mfw_: List[str] = []
        self.means_: np.ndarray = np.array([])
        self.stds_: np.ndarray = np.array([])
        self.doc_z_matrix_: np.ndarray = np.array([])
        self.doc_raw_matrix_: np.ndarray = np.array([])
        self.doc_authors_: List[str] = []

    def _vectorize(self, text_or_tokens: str | Sequence[str]) -> np.ndarray:
        toks = (
            tokenize_words(text_or_tokens)
            if isinstance(text_or_tokens, str)
            else [w.lower() for w in text_or_tokens]
        )
        n = max(1, len(toks))
        counts = Counter(toks)
        return np.array([counts.get(w, 0) / float(n) for w in self.mfw_], dtype=float)

    def fit(
        self,
        texts: Sequence[str | Sequence[str]],
        authors: Sequence[str],
    ) -> "GeneralImpostersVerifier":
        tokenized_docs = [
            tokenize_words(t) if isinstance(t, str) else [w.lower() for w in t]
            for t in texts
        ]
        self.doc_authors_ = list(authors)
        self.mfw_ = select_mfw(tokenized_docs, n_mfw=self.n_mfw)
        self.doc_raw_matrix_ = np.vstack([self._vectorize(t) for t in tokenized_docs])
        self.means_ = np.mean(self.doc_raw_matrix_, axis=0)
        self.stds_ = np.std(self.doc_raw_matrix_, axis=0, ddof=1)
        self.stds_[self.stds_ < 1e-12] = 1.0
        self.doc_z_matrix_ = (self.doc_raw_matrix_ - self.means_) / self.stds_
        return self

    def _subspace_distances(
        self,
        test_vec: np.ndarray,
        ref_mat: np.ndarray,
        feat_idx: np.ndarray,
    ) -> np.ndarray:
        sub_test = test_vec[feat_idx]
        sub_ref = ref_mat[:, feat_idx]

        if self.distance == "cosine":
            t_norm = float(np.linalg.norm(sub_test))
            if t_norm < 1e-12:
                return np.ones(len(sub_ref), dtype=float)
            r_norms = np.linalg.norm(sub_ref, axis=1)
            r_norms[r_norms < 1e-12] = 1.0
            sims = np.dot(sub_ref, sub_test) / (r_norms * t_norm)
            return 1.0 - sims

        elif self.distance == "minmax":
            mins = np.sum(np.minimum(sub_ref, sub_test), axis=1)
            maxs = np.sum(np.maximum(sub_ref, sub_test), axis=1)
            maxs[maxs < 1e-12] = 1.0
            return 1.0 - (mins / maxs)

        else:  # manhattan
            return np.mean(np.abs(sub_ref - sub_test), axis=1)

    def verify_all_candidates(
        self,
        text_or_tokens: str | Sequence[str],
    ) -> Dict[str, float]:
        """
        Compute the General Imposters verification score S(X, C) in [0.0, 1.0]
        for every candidate author C in the reference corpus.
        """
        rng = np.random.default_rng(self.random_state)
        raw_test = self._vectorize(text_or_tokens)
        z_test = (raw_test - self.means_) / self.stds_

        use_raw = self.distance == "minmax"
        test_vec = raw_test if use_raw else z_test
        ref_mat = self.doc_raw_matrix_ if use_raw else self.doc_z_matrix_

        n_feats = len(self.mfw_)
        k_feats = max(5, int(round(self.feature_sample_ratio * n_feats)))

        unique_authors = sorted(set(self.doc_authors_))
        author_indices = {
            a: np.array([i for i, doc_a in enumerate(self.doc_authors_) if doc_a == a], dtype=int)
            for a in unique_authors
        }
        imposter_pool_indices = {
            a: np.array([i for i, doc_a in enumerate(self.doc_authors_) if doc_a != a], dtype=int)
            for a in unique_authors
        }

        wins: Dict[str, int] = {a: 0 for a in unique_authors}

        for _ in range(self.n_iterations):
            feat_idx = rng.choice(n_feats, size=k_feats, replace=False)
            all_dists = self._subspace_distances(test_vec, ref_mat, feat_idx)

            for a in unique_authors:
                cand_idx = author_indices[a]
                imp_pool = imposter_pool_indices[a]
                if len(cand_idx) == 0 or len(imp_pool) == 0:
                    continue

                k_imp = max(1, int(round(self.imposter_sample_ratio * len(imp_pool))))
                sampled_imp_idx = rng.choice(imp_pool, size=k_imp, replace=False)

                # Compare candidate's top-k closest documents (or centroid/min distance)
                # against the closest imposter document
                cand_dist = float(np.min(all_dists[cand_idx]))
                imp_dist = float(np.min(all_dists[sampled_imp_idx]))

                if cand_dist < imp_dist:
                    wins[a] += 1

        return {a: wins[a] / float(self.n_iterations) for a in unique_authors}

    def predict_one(
        self, text_or_tokens: str | Sequence[str]
    ) -> Tuple[str, Dict[str, float]]:
        scores = self.verify_all_candidates(text_or_tokens)
        best_author = max(scores, key=scores.get)
        return best_author, scores

    def predict(self, texts: Sequence[str | Sequence[str]]) -> List[str]:
        return [self.predict_one(t)[0] for t in texts]
