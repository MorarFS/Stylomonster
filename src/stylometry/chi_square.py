"""
Kilgarriff's (2001) Chi-Squared (χ²) Method for Corpus Similarity and
Authorship Attribution.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from stylometry.preprocessing import select_mfw, tokenize_words


class KilgarriffChiSquare:
    """
    Implements Adam Kilgarriff's (2001) Chi-Squared stylometric distance.

    For each candidate author's pooled subcorpus A and a disputed document B:
    1. Select the top `n_mfw` most frequent words across their joint vocabulary
       (or across the reference corpus).
    2. Compute expected frequencies E_{w,A} and E_{w,B} from the joint distribution:
         joint_rate(w) = (C_A(w) + C_B(w)) / (N_A + N_B)
         E_{w,A} = joint_rate(w) * N_A
         E_{w,B} = joint_rate(w) * N_B
    3. Sum the Pearson chi-squared contributions over all MFWs:
         chi^2(A, B) = sum_{w} [ (C_A(w) - E_{w,A})^2 / E_{w,A} + (C_B(w) - E_{w,B})^2 / E_{w,B} ]
    Smaller chi^2 indicates greater stylistic homogeneity between the author and test text.
    """

    def __init__(self, n_mfw: int = 500, lowercase: bool = True):
        self.n_mfw = n_mfw
        self.lowercase = lowercase
        self.author_counters_: Dict[str, Counter[str]] = {}
        self.author_totals_: Dict[str, int] = {}
        self.reference_mfw_: List[str] = []

    def _tokenize(self, text_or_tokens: str | Sequence[str]) -> List[str]:
        if isinstance(text_or_tokens, str):
            return tokenize_words(text_or_tokens, lowercase=self.lowercase)
        return [w.lower() if self.lowercase else w for w in text_or_tokens]

    def fit(
        self,
        texts: Sequence[str | Sequence[str]],
        authors: Sequence[str],
    ) -> "KilgarriffChiSquare":
        author_tokens: Dict[str, List[str]] = {}
        for doc, author in zip(texts, authors):
            toks = self._tokenize(doc)
            author_tokens.setdefault(author, []).extend(toks)

        self.author_counters_ = {
            author: Counter(toks) for author, toks in author_tokens.items()
        }
        self.author_totals_ = {
            author: len(toks) for author, toks in author_tokens.items()
        }
        self.reference_mfw_ = select_mfw(
            list(author_tokens.values()), n_mfw=self.n_mfw
        )
        return self

    def chi_square_distance(
        self,
        author: str,
        test_tokens: Sequence[str],
    ) -> float:
        c_auth = self.author_counters_[author]
        n_auth = self.author_totals_[author]
        c_test = Counter(test_tokens)
        n_test = len(test_tokens)

        if n_auth == 0 or n_test == 0:
            return float("inf")

        joint_total = float(n_auth + n_test)
        # Joint top-N words for this comparison (Programming Historian / Kilgarriff 2001)
        joint_counter = c_auth + c_test
        top_words = [w for w, _ in joint_counter.most_common(self.n_mfw)]

        chi2 = 0.0
        for w in top_words:
            obs_auth = float(c_auth.get(w, 0))
            obs_test = float(c_test.get(w, 0))
            joint_count = obs_auth + obs_test
            if joint_count <= 0:
                continue

            exp_auth = joint_count * (n_auth / joint_total)
            exp_test = joint_count * (n_test / joint_total)

            if exp_auth > 0:
                chi2 += ((obs_auth - exp_auth) ** 2) / exp_auth
            if exp_test > 0:
                chi2 += ((obs_test - exp_test) ** 2) / exp_test

        return float(chi2)

    def predict_one(
        self, text_or_tokens: str | Sequence[str]
    ) -> Tuple[str, Dict[str, float]]:
        toks = self._tokenize(text_or_tokens)
        scores = {
            author: self.chi_square_distance(author, toks)
            for author in self.author_counters_
        }
        best_author = min(scores, key=scores.get)
        return best_author, scores

    def predict(self, texts: Sequence[str | Sequence[str]]) -> List[str]:
        return [self.predict_one(t)[0] for t in texts]
