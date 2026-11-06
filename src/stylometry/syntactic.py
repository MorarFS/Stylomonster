"""
Syntactic complexity, sentence rhythm, punctuation habit fingerprints,
readability metrics, function word profiles, and character n-gram stylometry.
"""

from __future__ import annotations
from collections import Counter
import re
from typing import Dict, List, Sequence, Tuple
import numpy as np

from stylometry.preprocessing import normalize_text, tokenize_sentences, tokenize_words
from stylometry.resources import MOSTELLER_WALLACE_MARKERS, STANDARD_FUNCTION_WORDS

_VOWEL_GROUP_RE = re.compile(r"[aeiouyAEIOUY]+")


def count_syllables(word: str) -> int:
    """Estimate English syllable count for a word token."""
    w = word.lower().strip("'")
    if len(w) <= 3:
        return 1
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    elif w.endswith(("es", "ed")) and len(w) > 4:
        w = w[:-2]
    groups = _VOWEL_GROUP_RE.findall(w)
    return max(1, len(groups))


def compute_syntactic_metrics(text: str) -> Dict[str, float]:
    """
    Extract syntactic rhythm, punctuation density (per 1,000 word tokens),
    and classical readability indices from a document.

    Punctuation habits (especially semicolons, colons, dashes, and parentheses)
    are among the strongest syntactic discriminators in literary and political prose.
    """
    norm = normalize_text(text)
    words = tokenize_words(norm)
    sentences = tokenize_sentences(norm)

    n_words = len(words)
    n_sents = max(1, len(sentences))

    if n_words == 0:
        return {
            "sentence_count": 0.0,
            "mean_sentence_length": 0.0,
            "std_sentence_length": 0.0,
            "median_sentence_length": 0.0,
            "iqr_sentence_length": 0.0,
            "short_sentence_ratio": 0.0,
            "long_sentence_ratio": 0.0,
            "comma_rate_per_1k": 0.0,
            "semicolon_rate_per_1k": 0.0,
            "colon_rate_per_1k": 0.0,
            "dash_rate_per_1k": 0.0,
            "question_rate_per_1k": 0.0,
            "exclamation_rate_per_1k": 0.0,
            "paren_rate_per_1k": 0.0,
            "quote_rate_per_1k": 0.0,
            "mean_syllables_per_word": 0.0,
            "complex_word_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "gunning_fog_index": 0.0,
        }

    sent_lengths = np.array(
        [len(tokenize_words(s)) for s in sentences if tokenize_words(s)],
        dtype=float,
    )
    if len(sent_lengths) == 0:
        sent_lengths = np.array([float(n_words)], dtype=float)

    mean_sl = float(np.mean(sent_lengths))
    std_sl = float(np.std(sent_lengths))
    med_sl = float(np.median(sent_lengths))
    q75, q25 = np.percentile(sent_lengths, [75, 25])
    iqr_sl = float(q75 - q25)
    short_ratio = float(np.mean(sent_lengths <= 12))
    long_ratio = float(np.mean(sent_lengths >= 35))

    # Punctuation rates per 1,000 word tokens
    scale = 1000.0 / float(n_words)
    comma_rate = norm.count(",") * scale
    semicolon_rate = norm.count(";") * scale
    colon_rate = norm.count(":") * scale
    dash_rate = (norm.count("\u2014") + norm.count(" - ")) * scale
    question_rate = norm.count("?") * scale
    exclamation_rate = norm.count("!") * scale
    paren_rate = (norm.count("(") + norm.count(")")) * scale
    quote_rate = norm.count('"') * scale

    # Syllable & character counts for readability indices
    syllables = [count_syllables(w) for w in words]
    total_syllables = sum(syllables)
    complex_words = sum(1 for s in syllables if s >= 3)
    total_chars = sum(len(w) for w in words)

    mean_syl = total_syllables / float(n_words)
    complex_ratio = complex_words / float(n_words)

    # Readability formulas
    flesch_ease = 206.835 - 1.015 * mean_sl - 84.6 * mean_syl
    fk_grade = 0.39 * mean_sl + 11.8 * mean_syl - 15.59
    L_per_100 = (total_chars / float(n_words)) * 100.0
    S_per_100 = (len(sent_lengths) / float(n_words)) * 100.0
    coleman_liau = 0.0588 * L_per_100 - 0.296 * S_per_100 - 15.8
    ari = 4.71 * (total_chars / float(n_words)) + 0.5 * mean_sl - 21.43
    gunning_fog = 0.4 * (mean_sl + 100.0 * complex_ratio)

    return {
        "sentence_count": float(len(sent_lengths)),
        "mean_sentence_length": mean_sl,
        "std_sentence_length": std_sl,
        "median_sentence_length": med_sl,
        "iqr_sentence_length": iqr_sl,
        "short_sentence_ratio": short_ratio,
        "long_sentence_ratio": long_ratio,
        "comma_rate_per_1k": float(comma_rate),
        "semicolon_rate_per_1k": float(semicolon_rate),
        "colon_rate_per_1k": float(colon_rate),
        "dash_rate_per_1k": float(dash_rate),
        "question_rate_per_1k": float(question_rate),
        "exclamation_rate_per_1k": float(exclamation_rate),
        "paren_rate_per_1k": float(paren_rate),
        "quote_rate_per_1k": float(quote_rate),
        "mean_syllables_per_word": float(mean_syl),
        "complex_word_ratio": float(complex_ratio),
        "flesch_reading_ease": float(flesch_ease),
        "flesch_kincaid_grade": float(fk_grade),
        "coleman_liau_index": float(coleman_liau),
        "automated_readability_index": float(ari),
        "gunning_fog_index": float(gunning_fog),
    }


def compute_function_word_rates(
    text_or_tokens: str | Sequence[str],
    word_list: Sequence[str] = STANDARD_FUNCTION_WORDS,
    per_n_tokens: float = 1000.0,
) -> Dict[str, float]:
    """
    Compute normalized rates (per `per_n_tokens` words) for a specified list
    of function words or Mosteller-Wallace markers.
    """
    tokens = (
        tokenize_words(text_or_tokens)
        if isinstance(text_or_tokens, str)
        else list(text_or_tokens)
    )
    n = len(tokens)
    if n == 0:
        return {w: 0.0 for w in word_list}
    counts = Counter(tokens)
    scale = per_n_tokens / float(n)
    return {w: counts.get(w, 0) * scale for w in word_list}


class CharNGramAnalyzer:
    """
    Character N-Gram Profile Attribution (Kešelj et al., 2003; Stamatatos, 2009).
    Extracts the top-K character n-grams (including word-boundary spaces and
    punctuation habits) and attributes authorship via Cosine Delta or Z-score
    Manhattan distance.
    """

    def __init__(self, n_values: Tuple[int, ...] = (3, 4), top_k: int = 500):
        self.n_values = n_values
        self.top_k = top_k
        self.vocab_: List[str] = []
        self.means_: np.ndarray = np.array([])
        self.stds_: np.ndarray = np.array([])
        self.author_centroids_: Dict[str, np.ndarray] = {}

    def _extract_ngrams(self, text: str) -> Counter[str]:
        norm = " ".join(normalize_text(text).lower().split())
        counts: Counter[str] = Counter()
        for n in self.n_values:
            if len(norm) >= n:
                counts.update(norm[i : i + n] for i in range(len(norm) - n + 1))
        return counts

    def _vectorize(self, text: str) -> np.ndarray:
        counts = self._extract_ngrams(text)
        total = max(1, sum(counts.values()))
        return np.array([counts.get(ng, 0) / float(total) for ng in self.vocab_], dtype=float)

    def fit(self, texts: Sequence[str], authors: Sequence[str]) -> "CharNGramAnalyzer":
        total_counts: Counter[str] = Counter()
        doc_counters = []
        for t in texts:
            c = self._extract_ngrams(t)
            total_counts.update(c)
            doc_counters.append(c)

        self.vocab_ = [ng for ng, _ in total_counts.most_common(self.top_k)]
        mat = np.zeros((len(texts), len(self.vocab_)), dtype=float)
        for i, c in enumerate(doc_counters):
            tot = max(1, sum(c.values()))
            for j, ng in enumerate(self.vocab_):
                mat[i, j] = c.get(ng, 0) / float(tot)

        self.means_ = np.mean(mat, axis=0)
        self.stds_ = np.std(mat, axis=0, ddof=1)
        self.stds_[self.stds_ < 1e-12] = 1.0

        z_mat = (mat - self.means_) / self.stds_
        self.author_centroids_ = {}
        for author in sorted(set(authors)):
            idx = [i for i, a in enumerate(authors) if a == author]
            centroid = np.mean(z_mat[idx], axis=0)
            norm = np.linalg.norm(centroid)
            self.author_centroids_[author] = centroid / norm if norm > 1e-12 else centroid

        return self

    def predict_one(self, text: str) -> Tuple[str, Dict[str, float]]:
        vec = self._vectorize(text)
        z = (vec - self.means_) / self.stds_
        z_norm = np.linalg.norm(z)
        z_unit = z / z_norm if z_norm > 1e-12 else z

        distances: Dict[str, float] = {}
        for author, centroid in self.author_centroids_.items():
            # Cosine distance in [0, 2]
            cos_sim = float(np.dot(z_unit, centroid))
            distances[author] = float(1.0 - cos_sim)
        best = min(distances, key=distances.get)
        return best, distances

    def predict(self, texts: Sequence[str]) -> List[str]:
        return [self.predict_one(t)[0] for t in texts]
