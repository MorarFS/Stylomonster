"""
The Delta Family of Stylometric Distance Measures:
- Burrows's Classic Delta (Manhattan Z-Score Distance, Burrows 2002)
- Cosine Delta / Würzburg Delta (Angular Z-Score Distance, Smith & Aldridge 2011; Evert et al. 2017)
- Argamon's Quadratic Delta (Euclidean Z-Score Distance, Argamon 2008)
- Eder's Delta (Rank-Weighted Manhattan Z-Score Distance, Eder et al. 2016)
- Canberra and MinMax (Ruzicka) Distances
Includes per-feature explainability to inspect which words drive attribution.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple
import numpy as np

from stylometry.preprocessing import select_mfw, tokenize_words

DeltaMetric = Literal["burrows", "cosine", "quadratic", "eder", "canberra", "minmax"]


def compute_delta_vector_distance(
    z_a: np.ndarray,
    z_b: np.ndarray,
    metric: DeltaMetric = "burrows",
    raw_a: Optional[np.ndarray] = None,
    raw_b: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the stylometric distance between two feature profiles `z_a` and `z_b`.
    """
    m = len(z_a)
    if m == 0:
        return 0.0

    if metric == "burrows":
        # Burrows (2002): Mean absolute difference of z-scores
        return float(np.mean(np.abs(z_a - z_b)))

    elif metric == "quadratic":
        # Argamon (2008): Root-mean-square Euclidean distance of z-scores
        return float(np.sqrt(np.mean((z_a - z_b) ** 2)))

    elif metric == "cosine":
        # Smith & Aldridge (2011); Evert et al. (2017) Würzburg Cosine Delta:
        # 1 - cosine_similarity(z_a, z_b)
        norm_a = float(np.linalg.norm(z_a))
        norm_b = float(np.linalg.norm(z_b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 1.0
        cos_sim = float(np.dot(z_a, z_b) / (norm_a * norm_b))
        return float(max(0.0, 1.0 - cos_sim))

    elif metric == "eder":
        # Eder et al. (2016): Rank-weighted Manhattan z-score distance
        # Weight w_i = (M - i + 1) / M for i = 1..M
        ranks = np.arange(1, m + 1, dtype=float)
        weights = (m - ranks + 1.0) / float(m)
        weights /= np.sum(weights)
        return float(np.sum(weights * np.abs(z_a - z_b)))

    elif metric == "canberra":
        v_a = raw_a if raw_a is not None else np.abs(z_a)
        v_b = raw_b if raw_b is not None else np.abs(z_b)
        denom = np.abs(v_a) + np.abs(v_b) + 1e-12
        return float(np.mean(np.abs(v_a - v_b) / denom))

    elif metric == "minmax":
        # Ruzicka / MinMax distance used in Koppel's Imposters and Kestemont et al. (2016)
        v_a = raw_a if raw_a is not None else np.maximum(0.0, z_a + 3.0)
        v_b = raw_b if raw_b is not None else np.maximum(0.0, z_b + 3.0)
        mins = np.sum(np.minimum(v_a, v_b))
        maxs = np.sum(np.maximum(v_a, v_b))
        if maxs < 1e-12:
            return 1.0
        return float(1.0 - (mins / maxs))

    else:
        raise ValueError(f"Unknown Delta metric: {metric}")


class DeltaAnalyzer:
    """
    Unified analyzer for Burrows's Delta, Cosine (Würzburg) Delta, Quadratic
    (Argamon) Delta, Eder's Delta, Canberra, and MinMax distances.

    Supports two standard reference profile modes:
    - `profile_mode="author_subcorpus"`: Pools each author's known texts into an
      authorial subcorpus profile (as in Burrows 2002 & Programming Historian),
      ideal when individual training texts vary in length.
    - `profile_mode="document_centroid"`: Computes corpus mean & std across all
      individual training documents and represents each author by the mean z-vector
      of their documents (as in Evert et al. 2017).
    """

    def __init__(
        self,
        n_mfw: int = 150,
        metric: DeltaMetric = "cosine",
        profile_mode: Literal["author_subcorpus", "document_centroid"] = "document_centroid",
        culling_ratio: float = 0.0,
        remove_pronouns: bool = False,
        allowed_vocab: Optional[Set[str]] = None,
    ):
        self.n_mfw = n_mfw
        self.metric = metric
        self.profile_mode = profile_mode
        self.culling_ratio = culling_ratio
        self.remove_pronouns = remove_pronouns
        self.allowed_vocab = allowed_vocab

        self.mfw_: List[str] = []
        self.means_: np.ndarray = np.array([])
        self.stds_: np.ndarray = np.array([])
        self.author_z_profiles_: Dict[str, np.ndarray] = {}
        self.author_raw_profiles_: Dict[str, np.ndarray] = {}
        self.doc_z_matrix_: np.ndarray = np.array([])
        self.doc_raw_matrix_: np.ndarray = np.array([])
        self.doc_authors_: List[str] = []

    def _tokenize(self, text_or_tokens: str | Sequence[str]) -> List[str]:
        if isinstance(text_or_tokens, str):
            return tokenize_words(
                text_or_tokens,
                lowercase=True,
                remove_pronouns=self.remove_pronouns,
            )
        toks = [w.lower() for w in text_or_tokens]
        if self.remove_pronouns:
            from stylometry.resources import PERSONAL_PRONOUNS
            toks = [w for w in toks if w not in PERSONAL_PRONOUNS]
        return toks

    def _relative_freq_vector(self, tokens: Sequence[str]) -> np.ndarray:
        n = len(tokens)
        if n == 0 or not self.mfw_:
            return np.zeros(len(self.mfw_), dtype=float)
        counts = Counter(tokens)
        return np.array([counts.get(w, 0) / float(n) for w in self.mfw_], dtype=float)

    def fit(
        self,
        texts: Sequence[str | Sequence[str]],
        authors: Sequence[str],
    ) -> "DeltaAnalyzer":
        tokenized_docs = [self._tokenize(t) for t in texts]
        self.doc_authors_ = list(authors)
        unique_authors = sorted(set(authors))

        # Select MFW across the training documents
        self.mfw_ = select_mfw(
            tokenized_docs,
            n_mfw=self.n_mfw,
            culling_ratio=self.culling_ratio,
            allowed_vocab=self.allowed_vocab,
        )
        self.doc_raw_matrix_ = np.vstack(
            [self._relative_freq_vector(t) for t in tokenized_docs]
        )
        self.means_ = np.mean(self.doc_raw_matrix_, axis=0)
        ddof = 1 if len(tokenized_docs) > 1 else 0
        self.stds_ = np.std(self.doc_raw_matrix_, axis=0, ddof=ddof)
        self.stds_[self.stds_ < 1e-12] = 1.0

        self.doc_z_matrix_ = (self.doc_raw_matrix_ - self.means_) / self.stds_

        if self.profile_mode == "author_subcorpus":
            # Pool tokens per author for each author's reference profile,
            # standardized by the corpus document-level mean & std
            author_tokens: Dict[str, List[str]] = {a: [] for a in unique_authors}
            for toks, a in zip(tokenized_docs, authors):
                author_tokens[a].extend(toks)

            for a in unique_authors:
                raw_prof = self._relative_freq_vector(author_tokens[a])
                self.author_raw_profiles_[a] = raw_prof
                self.author_z_profiles_[a] = (raw_prof - self.means_) / self.stds_
        else:
            # Document centroid mode: mean z-vector across each author's documents
            for a in unique_authors:
                idx = [i for i, doc_a in enumerate(authors) if doc_a == a]
                self.author_raw_profiles_[a] = np.mean(self.doc_raw_matrix_[idx], axis=0)
                self.author_z_profiles_[a] = np.mean(self.doc_z_matrix_[idx], axis=0)

        return self

    def transform_one(self, text_or_tokens: str | Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(z_vector, raw_rel_freq_vector)` for a document."""
        toks = self._tokenize(text_or_tokens)
        raw = self._relative_freq_vector(toks)
        z = (raw - self.means_) / self.stds_
        return z, raw

    def distances_to_authors(
        self,
        text_or_tokens: str | Sequence[str],
        metric: Optional[DeltaMetric] = None,
    ) -> Dict[str, float]:
        """Compute Delta distances from a test document to every candidate author."""
        active_metric = metric or self.metric
        z_test, raw_test = self.transform_one(text_or_tokens)
        distances: Dict[str, float] = {}
        for author, z_auth in self.author_z_profiles_.items():
            raw_auth = self.author_raw_profiles_[author]
            distances[author] = compute_delta_vector_distance(
                z_test, z_auth, metric=active_metric, raw_a=raw_test, raw_b=raw_auth
            )
        return distances

    def predict_one(
        self,
        text_or_tokens: str | Sequence[str],
        metric: Optional[DeltaMetric] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Predict the author with minimum Delta distance and return all distances."""
        dists = self.distances_to_authors(text_or_tokens, metric=metric)
        best_author = min(dists, key=dists.get)
        return best_author, dists

    def predict(
        self,
        texts: Sequence[str | Sequence[str]],
        metric: Optional[DeltaMetric] = None,
    ) -> List[str]:
        return [self.predict_one(t, metric=metric)[0] for t in texts]

    def explain_attribution(
        self,
        text_or_tokens: str | Sequence[str],
        author_a: str,
        author_b: str,
        top_n: int = 12,
    ) -> List[Dict[str, object]]:
        """
        Identify the top MFW features that most strongly pull the test document
        toward `author_a` vs. `author_b` based on z-score proximity.
        """
        z_test, _ = self.transform_one(text_or_tokens)
        z_a = self.author_z_profiles_[author_a]
        z_b = self.author_z_profiles_[author_b]

        contributions = []
        for i, word in enumerate(self.mfw_):
            dist_a = abs(float(z_test[i] - z_a[i]))
            dist_b = abs(float(z_test[i] - z_b[i]))
            # Positive advantage means test document is closer to author_a than author_b on this word
            advantage_a = dist_b - dist_a
            contributions.append(
                {
                    "word": word,
                    "rank": i + 1,
                    "test_z": round(float(z_test[i]), 3),
                    f"{author_a}_z": round(float(z_a[i]), 3),
                    f"{author_b}_z": round(float(z_b[i]), 3),
                    f"favors": author_a if advantage_a >= 0 else author_b,
                    "abs_advantage": round(abs(advantage_a), 4),
                    "signed_advantage_for_" + author_a: round(advantage_a, 4),
                }
            )
        contributions.sort(key=lambda x: x["abs_advantage"], reverse=True)
        return contributions[:top_n]
