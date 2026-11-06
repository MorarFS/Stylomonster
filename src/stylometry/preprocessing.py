"""
Text preprocessing, tokenization, vocabulary culling, and document segmentation
utilities for computational stylometry.
"""

from __future__ import annotations
from collections import Counter
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from stylometry.resources import PERSONAL_PRONOUNS

# Fast regex patterns for stylometric tokenization
_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
_TOKEN_WITH_PUNCT_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[.,;:!?\-\u2014()\"']")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n\s*\n")


def normalize_text(text: str) -> str:
    """Normalize line endings, dashes, and curly quotes for uniform tokenization."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Unify typographic quotes and apostrophes
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    # Normalize double-hyphens to em-dash
    text = re.sub(r"--+", "\u2014", text)
    return text


def tokenize_words(
    text: str,
    lowercase: bool = True,
    remove_pronouns: bool = False,
    exclude_words: Optional[Set[str]] = None,
) -> List[str]:
    """
    Extract alphabetic word tokens from text.

    Parameters
    ----------
    text : str
        Raw input text.
    lowercase : bool
        If True (default), case-fold all tokens to lowercase.
    remove_pronouns : bool
        If True, filter out personal pronouns (common in literary stylometry to
        reduce narrative perspective / character gender bias).
    exclude_words : Optional[Set[str]]
        Additional set of lowercase words to exclude.
    """
    norm = normalize_text(text)
    words = _WORD_RE.findall(norm)
    if lowercase:
        words = [w.lower() for w in words]
    if remove_pronouns:
        words = [w for w in words if w.lower() not in PERSONAL_PRONOUNS]
    if exclude_words:
        words = [w for w in words if w.lower() not in exclude_words]
    return words


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation and paragraph boundaries."""
    norm = normalize_text(text).strip()
    if not norm:
        return []
    raw_sents = _SENTENCE_SPLIT_RE.split(norm)
    sentences = []
    for s in raw_sents:
        cleaned = " ".join(s.split())
        if cleaned and _WORD_RE.search(cleaned):
            sentences.append(cleaned)
    return sentences


def select_mfw(
    tokenized_docs: Sequence[Sequence[str]],
    n_mfw: int = 150,
    culling_ratio: float = 0.0,
    allowed_vocab: Optional[Set[str]] = None,
) -> List[str]:
    """
    Select the Top-N Most Frequent Words (MFW) across a corpus, with optional
    Hoover (2004) / Eder (2016) vocabulary culling.

    Parameters
    ----------
    tokenized_docs : Sequence[Sequence[str]]
        List of word token sequences for each document in the reference corpus.
    n_mfw : int
        Number of most frequent words to retain.
    culling_ratio : float
        Minimum fraction of documents (0.0 to 1.0) in which a word must appear
        to be eligible for the MFW list. E.g. 0.5 means the word must appear in
        at least 50% of corpus documents.
    allowed_vocab : Optional[Set[str]]
        If provided, restrict candidate words to this vocabulary (e.g., function words).
    """
    if not tokenized_docs:
        return []

    total_counts: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    n_docs = len(tokenized_docs)

    for tokens in tokenized_docs:
        total_counts.update(tokens)
        doc_freq.update(set(tokens))

    min_docs = max(1, int(round(culling_ratio * n_docs))) if culling_ratio > 0 else 1

    candidates: List[Tuple[str, int]] = []
    for word, count in total_counts.most_common():
        if allowed_vocab is not None and word not in allowed_vocab:
            continue
        if doc_freq[word] < min_docs:
            continue
        candidates.append((word, count))
        if len(candidates) >= n_mfw:
            break

    return [w for w, _ in candidates]


def chunk_tokens(
    tokens: Sequence[str],
    chunk_size: int = 2000,
    step_size: Optional[int] = None,
    drop_Incomplete: bool = True,
) -> List[List[str]]:
    """
    Segment a long token sequence into fixed-length windows (for Rolling Stylometry
    or equal-length novel chapter comparisons).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    step = step_size or chunk_size
    chunks: List[List[str]] = []
    for start in range(0, len(tokens), step):
        window = list(tokens[start : start + chunk_size])
        if len(window) < chunk_size and drop_Incomplete:
            break
        if window:
            chunks.append(window)
    return chunks
