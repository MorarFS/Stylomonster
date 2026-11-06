"""
Lexical richness, vocabulary diversity metrics, and Mendenhall's (1887)
Characteristic Curves of Composition.
"""

from __future__ import annotations
from collections import Counter
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from stylometry.preprocessing import tokenize_words


def compute_mattr(tokens: Sequence[str], window_size: int = 500) -> float:
    """
    Compute the Moving-Average Type-Token Ratio (MATTR; Covington & McFall, 2010),
    which measures lexical diversity completely independent of total text length.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window_size:
        return len(set(tokens)) / float(n)

    counts: Counter[str] = Counter(tokens[:window_size])
    unique_counts = [len(counts)]

    for i in range(window_size, n):
        out_tok = tokens[i - window_size]
        in_tok = tokens[i]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
        counts[in_tok] += 1
        unique_counts.append(len(counts))

    return float(np.mean(unique_counts)) / float(window_size)


def compute_lexical_metrics(
    text_or_tokens: str | Sequence[str],
    mattr_window: int = 500,
) -> Dict[str, float]:
    """
    Compute a comprehensive suite of classical and modern lexical richness
    and vocabulary diversity statistics:

    - `token_count` (N): Total word tokens
    - `type_count` (V): Unique word types
    - `mean_word_length`: Average characters per word token
    - `std_word_length`: Standard deviation of word lengths
    - `ttr`: Raw Type-Token Ratio (V / N)
    - `root_ttr_guiraud`: Guiraud's (1954) Root TTR = V / sqrt(N)
    - `log_ttr_herdan`: Herdan's (1960) C = log(V) / log(N)
    - `mattr`: Moving-Average TTR (Covington & McFall, 2010)
    - `hapax_ratio`: Ratio of hapax legomena (V_1, words occurring once) to V
    - `dislegomena_ratio`: Sichel's (1975) S = V_2 / V (words occurring twice)
    - `yules_k`: Yule's (1944) K Characteristic = 10^4 * (sum(i^2 * V_i) - N) / N^2
      (lower K indicates richer vocabulary; higher K indicates higher repetition)
    - `simpsons_d`: Simpson's (1949) Diversity Index D = sum(n_w * (n_w - 1)) / (N * (N - 1))
    - `honores_r`: Honoré's (1979) Statistic R = 100 * ln(N) / (1 - V_1 / V)
    - `brunets_w`: Brunét's (1978) Index W = N^(V^(-0.165))
    """
    tokens = (
        tokenize_words(text_or_tokens)
        if isinstance(text_or_tokens, str)
        else list(text_or_tokens)
    )
    n = len(tokens)
    if n == 0:
        return {
            "token_count": 0.0,
            "type_count": 0.0,
            "mean_word_length": 0.0,
            "std_word_length": 0.0,
            "ttr": 0.0,
            "root_ttr_guiraud": 0.0,
            "log_ttr_herdan": 0.0,
            "mattr": 0.0,
            "hapax_ratio": 0.0,
            "dislegomena_ratio": 0.0,
            "yules_k": 0.0,
            "simpsons_d": 0.0,
            "honores_r": 0.0,
            "brunets_w": 0.0,
        }

    lengths = np.array([len(w) for w in tokens], dtype=float)
    word_counts = Counter(tokens)
    v = len(word_counts)

    # Spectrum of frequencies: V_i = number of word types that occur exactly i times
    spectrum = Counter(word_counts.values())
    v1 = spectrum.get(1, 0)
    v2 = spectrum.get(2, 0)

    ttr = v / float(n)
    root_ttr = v / math.sqrt(n)
    log_ttr = math.log(v) / math.log(n) if n > 1 and v > 1 else 1.0
    mattr = compute_mattr(tokens, window_size=mattr_window)

    hapax_ratio = v1 / float(v)
    dislegomena_ratio = v2 / float(v)

    # Yule's K Characteristic (1944)
    s2 = sum((freq ** 2) * count_types for freq, count_types in spectrum.items())
    yules_k = 10000.0 * (s2 - n) / float(n * n)

    # Simpson's D (1949)
    if n > 1:
        simpsons_d = sum(freq * (freq - 1) * count_types for freq, count_types in spectrum.items()) / float(n * (n - 1))
    else:
        simpsons_d = 0.0

    # Honoré's R (1979)
    denom = 1.0 - (v1 / float(v))
    if denom > 1e-6:
        honores_r = 100.0 * math.log(n) / denom
    else:
        honores_r = 100.0 * math.log(max(n, 2)) * v

    # Brunét's W (1978)
    brunets_w = float(n ** (v ** -0.165))

    return {
        "token_count": float(n),
        "type_count": float(v),
        "mean_word_length": float(np.mean(lengths)),
        "std_word_length": float(np.std(lengths)),
        "ttr": float(ttr),
        "root_ttr_guiraud": float(root_ttr),
        "log_ttr_herdan": float(log_ttr),
        "mattr": float(mattr),
        "hapax_ratio": float(hapax_ratio),
        "dislegomena_ratio": float(dislegomena_ratio),
        "yules_k": float(yules_k),
        "simpsons_d": float(simpsons_d),
        "honores_r": float(honores_r),
        "brunets_w": float(brunets_w),
    }


class MendenhallCurveAnalyzer:
    """
    Implements T. C. Mendenhall's (1887) 'Characteristic Curves of Composition'.
    Computes normalized word-length probability distributions P(|w| = k) for
    k in 1..max_length and attributes authorship via L1, L2, or Chi-Square
    curve distance.
    """

    def __init__(self, max_word_length: int = 16):
        self.max_word_length = max_word_length
        self.author_curves_: Dict[str, np.ndarray] = {}

    def compute_curve(self, text_or_tokens: str | Sequence[str]) -> np.ndarray:
        """Return a normalized vector of length `max_word_length` for word lengths 1..max_word_length."""
        tokens = (
            tokenize_words(text_or_tokens)
            if isinstance(text_or_tokens, str)
            else list(text_or_tokens)
        )
        curve = np.zeros(self.max_word_length, dtype=float)
        if not tokens:
            return curve
        for w in tokens:
            idx = min(len(w), self.max_word_length) - 1
            if idx >= 0:
                curve[idx] += 1.0
        curve /= float(len(tokens))
        return curve

    def fit(self, texts: Sequence[str | Sequence[str]], authors: Sequence[str]) -> "MendenhallCurveAnalyzer":
        """Compute pooled characteristic word-length curves for each candidate author."""
        grouped: Dict[str, List[str]] = {}
        for doc, author in zip(texts, authors):
            toks = tokenize_words(doc) if isinstance(doc, str) else list(doc)
            grouped.setdefault(author, []).extend(toks)

        self.author_curves_ = {
            author: self.compute_curve(toks) for author, toks in grouped.items()
        }
        return self

    def curve_distance(self, curve_a: np.ndarray, curve_b: np.ndarray, metric: str = "l1") -> float:
        """Compute distance between two Mendenhall composition curves."""
        if metric == "l1":
            return float(np.sum(np.abs(curve_a - curve_b)))
        elif metric == "l2":
            return float(np.sqrt(np.sum((curve_a - curve_b) ** 2)))
        elif metric == "chi2":
            denom = curve_a + curve_b + 1e-12
            return float(0.5 * np.sum(((curve_a - curve_b) ** 2) / denom))
        else:
            raise ValueError(f"Unsupported curve metric: {metric}")

    def predict_one(
        self, text_or_tokens: str | Sequence[str], metric: str = "l1"
    ) -> Tuple[str, Dict[str, float]]:
        """Predict closest author and return dictionary of curve distances."""
        test_curve = self.compute_curve(text_or_tokens)
        distances = {
            author: self.curve_distance(test_curve, author_curve, metric=metric)
            for author, author_curve in self.author_curves_.items()
        }
        best_author = min(distances, key=distances.get)
        return best_author, distances

    def predict(
        self, texts: Sequence[str | Sequence[str]], metric: str = "l1"
    ) -> List[str]:
        return [self.predict_one(t, metric=metric)[0] for t in texts]
