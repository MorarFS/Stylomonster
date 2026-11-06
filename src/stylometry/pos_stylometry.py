"""
Morphosyntactic & Part-of-Speech (POS) Tag N-Gram Stylometry (Hirst & Feiguina,
2007; Stamatatos, 2009).

Provides a fast, deterministic, zero-external-corpus morphosyntactic tagger that
maps every token into 16 fine-grained grammatical & morphological categories
(Preposition, Determiner, Coordinating Conjunction, Subordinating Conjunction,
Modal/Auxiliary, Pronoun, Negation, Discourse Adverb, Nominal Suffix, Verbal
Suffix, Adjectival Suffix, Adverbial `-ly`, Punctuation Clause Boundary, etc.)
and models syntactic sequence transitions via POS 1-grams, 2-grams, and 3-grams.
"""

from __future__ import annotations
from collections import Counter
import re
from typing import Dict, List, Sequence, Tuple
import numpy as np

from stylometry.preprocessing import normalize_text

_TOKEN_PUNCT_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[.,;:!?\-\u2014()]")

# Closed-class grammatical categories for deterministic morphosyntactic tagging
_DET = {"the", "a", "an", "this", "that", "these", "those", "such", "each", "every", "either", "neither", "another", "some", "any", "all", "both", "several", "many", "much", "few", "enough"}
_PREP = {"of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "among", "amongst", "upon", "unto", "within", "towards", "toward", "throughout", "despite", "beyond", "across", "behind", "beside", "besides", "above", "below", "since", "until", "till", "concerning", "respecting", "regarding"}
_CCONJ = {"and", "but", "or", "nor", "yet", "so"}
_SCONJ = {"although", "though", "because", "unless", "whereas", "while", "whilst", "whereby", "wherein", "whereof", "whether", "if", "than", "lest", "inasmuch", "insomuch", "provided"}
_PRON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "one", "oneself"}
_REL_WH = {"who", "whom", "whose", "which", "what", "where", "when", "why", "how", "whatever", "whoever", "whichever"}
_AUX_MOD = {"be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "having", "do", "does", "did", "done", "doing", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "ought", "need", "dare"}
_NEG = {"not", "no", "never", "neither", "none", "nobody", "nothing", "nowhere"}
_DISC_ADV = {"however", "therefore", "thus", "hence", "consequently", "accordingly", "nevertheless", "nonetheless", "moreover", "furthermore", "otherwise", "instead", "meanwhile", "indeed", "perhaps", "also", "too", "only", "merely", "simply", "even", "still", "already", "always", "often", "sometimes", "seldom", "commonly"}


def morphosyntactic_tag(token: str) -> str:
    """Assign one of 16 morphosyntactic tags to a word or punctuation token."""
    t = token.lower()
    if t in {",", ";", ":"}:
        return "P_CLAUSE"
    if t in {".", "!", "?"}:
        return "P_END"
    if t in {"-", "\u2014", "(", ")"}:
        return "P_PAREN"
    if t in _DET:
        return "DET"
    if t in _PREP:
        return "PREP"
    if t in _CCONJ:
        return "CCONJ"
    if t in _SCONJ:
        return "SCONJ"
    if t in _PRON:
        return "PRON"
    if t in _REL_WH:
        return "WH"
    if t in _AUX_MOD:
        return "AUX"
    if t in _NEG:
        return "NEG"
    if t in _DISC_ADV:
        return "DISC"
    if t.endswith("ly") and len(t) > 4:
        return "ADV_LY"
    if t.endswith(("tion", "sion", "ment", "ness", "ity", "ism", "ance", "ence", "ship", "hood")):
        return "NOUN_DERIV"
    if t.endswith(("ed", "ing", "ize", "ise", "ate", "fy")) and len(t) > 4:
        return "VERB_MORPH"
    if t.endswith(("able", "ible", "ous", "ive", "al", "ic", "ful", "less")) and len(t) > 4:
        return "ADJ_DERIV"
    return "LEX_ROOT"


def extract_pos_tag_sequence(text: str) -> List[str]:
    norm = normalize_text(text)
    tokens = _TOKEN_PUNCT_RE.findall(norm)
    return [morphosyntactic_tag(tok) for tok in tokens]


class POSTagNGramAnalyzer:
    """
    Attributes authorship using standardized frequencies of morphosyntactic / POS
    tag 1-grams, 2-grams, and 3-grams (Cosine Delta over syntactic transition patterns).
    """

    def __init__(self, n_values: Tuple[int, ...] = (1, 2, 3), top_k: int = 250):
        self.n_values = n_values
        self.top_k = top_k
        self.vocab_: List[str] = []
        self.means_: np.ndarray = np.array([])
        self.stds_: np.ndarray = np.array([])
        self.author_centroids_: Dict[str, np.ndarray] = {}

    def _count_pos_ngrams(self, text: str) -> Counter[str]:
        tags = extract_pos_tag_sequence(text)
        counts: Counter[str] = Counter()
        for n in self.n_values:
            if len(tags) >= n:
                counts.update(
                    "_".join(tags[i : i + n]) for i in range(len(tags) - n + 1)
                )
        return counts

    def fit(self, texts: Sequence[str], authors: Sequence[str]) -> "POSTagNGramAnalyzer":
        total: Counter[str] = Counter()
        doc_counters = []
        for t in texts:
            c = self._count_pos_ngrams(t)
            total.update(c)
            doc_counters.append(c)

        self.vocab_ = [g for g, _ in total.most_common(self.top_k)]
        mat = np.zeros((len(texts), len(self.vocab_)), dtype=float)
        for i, c in enumerate(doc_counters):
            tot = max(1, sum(c.values()))
            for j, g in enumerate(self.vocab_):
                mat[i, j] = c.get(g, 0) / float(tot)

        self.means_ = np.mean(mat, axis=0)
        self.stds_ = np.std(mat, axis=0, ddof=1)
        self.stds_[self.stds_ < 1e-12] = 1.0
        z_mat = (mat - self.means_) / self.stds_

        self.author_centroids_ = {}
        for a in sorted(set(authors)):
            idx = [i for i, doc_a in enumerate(authors) if doc_a == a]
            centroid = np.mean(z_mat[idx], axis=0)
            norm = np.linalg.norm(centroid)
            self.author_centroids_[a] = centroid / norm if norm > 1e-12 else centroid
        return self

    def predict_one(self, text: str) -> Tuple[str, Dict[str, float]]:
        c = self._count_pos_ngrams(text)
        tot = max(1, sum(c.values()))
        vec = np.array([c.get(g, 0) / float(tot) for g in self.vocab_], dtype=float)
        z = (vec - self.means_) / self.stds_
        norm = np.linalg.norm(z)
        z_unit = z / norm if norm > 1e-12 else z

        dists: Dict[str, float] = {}
        for a, centroid in self.author_centroids_.items():
            dists[a] = float(1.0 - np.dot(z_unit, centroid))
        best = min(dists, key=dists.get)
        return best, dists

    def predict(self, texts: Sequence[str]) -> List[str]:
        return [self.predict_one(t)[0] for t in texts]
