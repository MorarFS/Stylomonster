"""
Benchmark corpus parsers and loaders for:
1. The Federalist Papers (1787–1788) — all 85 essays from Project Gutenberg #1404
   with strict header/salutation/signature stripping and historical authorship labels.
2. 19th-Century Classic Literature Corpus — 15 novels across 5 Victorian/Edwardian
   authors (Jane Austen, Charlotte Brontë, Charles Dickens, Sir Arthur Conan Doyle,
   and H. G. Wells), with clean boilerplate removal and equal-sized segmentation.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence

from stylometry.preprocessing import tokenize_words


@dataclass
class CorpusDocument:
    doc_id: str
    title: str
    author: str
    status: str  # "undisputed", "disputed", "joint", "test_case"
    text: str
    word_count: int
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        return d


# Historical Ground-Truth Authorship Mapping of the 85 Federalist Papers
# (Mosteller & Wallace, 1964; Rossiter, 1961; Programming Historian):
# - Hamilton Undisputed (51): 1, 6-9, 11-13, 15-17, 21-36, 59-61, 65-85
# - Madison Undisputed (14): 10, 14, 37-48
# - Jay Undisputed (5): 2, 3, 4, 5, 64 (Note: #64 was historically verified via Jay's draft)
# - Joint (Hamilton & Madison, 3): 18, 19, 20
# - Disputed (Claimed by both Hamilton & Madison, 12): 49-58, 62, 63
HAMILTON_PAPERS = (
    {1}
    | set(range(6, 10))
    | set(range(11, 14))
    | set(range(15, 18))
    | set(range(21, 37))
    | set(range(59, 62))
    | set(range(65, 86))
)
MADISON_PAPERS = {10, 14} | set(range(37, 49))
JAY_PAPERS = {2, 3, 4, 5, 64}
JOINT_PAPERS = {18, 19, 20}
DISPUTED_PAPERS = set(range(49, 59)) | {62, 63}


def get_federalist_ground_truth(paper_num: int) -> Tuple[str, str]:
    """Return `(author, status)` for Federalist Paper number `1..85`."""
    if paper_num in HAMILTON_PAPERS:
        return ("Hamilton", "undisputed")
    if paper_num in MADISON_PAPERS:
        return ("Madison", "undisputed")
    if paper_num in JAY_PAPERS:
        return ("Jay", "undisputed")
    if paper_num in JOINT_PAPERS:
        return ("Joint (Hamilton & Madison)", "joint")
    if paper_num in DISPUTED_PAPERS:
        return ("Disputed (Hamilton or Madison)", "disputed")
    raise ValueError(f"Invalid Federalist paper number: {paper_num}")


def parse_federalist_corpus(raw_path: str | Path) -> List[CorpusDocument]:
    """
    Parse Project Gutenberg's The Federalist Papers (`pg1404.txt`) into 85 clean
    `CorpusDocument` objects, stripping all editorial headers, author lines,
    salutations ('To the People of the State of New York:'), and trailing 'PUBLIUS'
    signatures so that zero author metadata leaks into the text.
    """
    raw_text = Path(raw_path).read_text(encoding="utf-8")
    start_idx = raw_text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_idx = raw_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start_idx != -1 and end_idx != -1:
        body = raw_text[start_idx:end_idx]
    else:
        body = raw_text

    parts = re.split(r"\n(?=FEDERALIST No\.\s+\d+)", body)
    paper_blocks = [p for p in parts if re.match(r"^\s*FEDERALIST No\.\s+\d+", p)]

    if len(paper_blocks) != 85:
        raise RuntimeError(f"Expected 85 Federalist Papers, found {len(paper_blocks)}")

    documents: List[CorpusDocument] = []
    for block in paper_blocks:
        m_num = re.search(r"FEDERALIST No\.\s+(\d+)", block)
        if not m_num:
            continue
        paper_num = int(m_num.group(1))

        # Extract title from lines between FEDERALIST No. X and salutation
        salutation_match = re.search(
            r"To the People of the State of New York\s*[:.]?",
            block,
            flags=re.IGNORECASE,
        )
        if not salutation_match:
            raise RuntimeError(f"Missing salutation in Federalist No. {paper_num}")

        header_part = block[: salutation_match.start()]
        essay_body = block[salutation_match.end() :].strip()

        # Remove trailing PUBLIUS signature
        essay_body = re.sub(r"\bPUBLIUS\.?\s*$", "", essay_body, flags=re.MULTILINE).strip()

        header_lines = [
            l.strip()
            for l in header_part.splitlines()
            if l.strip()
            and not l.strip().startswith("FEDERALIST No.")
            and not re.match(r"^(HAMILTON|MADISON|JAY|HAMILTON OR MADISON|HAMILTON AND MADISON|MADISON, with HAMILTON)$", l.strip())
            and not re.match(r"^(For|From)\s+the\b", l.strip(), flags=re.IGNORECASE)
        ]
        title = " ".join(header_lines[:2]) if header_lines else f"Federalist No. {paper_num}"

        author, status = get_federalist_ground_truth(paper_num)
        words = tokenize_words(essay_body)

        documents.append(
            CorpusDocument(
                doc_id=f"fed_{paper_num:02d}",
                title=f"Federalist No. {paper_num}: {title}",
                author=author,
                status=status,
                text=essay_body,
                word_count=len(words),
                metadata={"paper_number": paper_num},
            )
        )

    documents.sort(key=lambda d: int(d.metadata["paper_number"]))
    return documents


NOVEL_METADATA: List[Dict[str, str]] = [
    {
        "file": "austen_pride.txt",
        "doc_id": "austen_pride",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "role": "train",
    },
    {
        "file": "austen_emma.txt",
        "doc_id": "austen_emma",
        "title": "Emma",
        "author": "Jane Austen",
        "role": "train",
    },
    {
        "file": "austen_sense.txt",
        "doc_id": "austen_sense",
        "title": "Sense and Sensibility",
        "author": "Jane Austen",
        "role": "test",
    },
    {
        "file": "bronte_jane_eyre.txt",
        "doc_id": "bronte_jane_eyre",
        "title": "Jane Eyre",
        "author": "Charlotte Brontë",
        "role": "train",
    },
    {
        "file": "bronte_shirley.txt",
        "doc_id": "bronte_shirley",
        "title": "Shirley",
        "author": "Charlotte Brontë",
        "role": "train",
    },
    {
        "file": "bronte_villette.txt",
        "doc_id": "bronte_villette",
        "title": "Villette",
        "author": "Charlotte Brontë",
        "role": "test",
    },
    {
        "file": "dickens_great_expectations.txt",
        "doc_id": "dickens_great_expectations",
        "title": "Great Expectations",
        "author": "Charles Dickens",
        "role": "train",
    },
    {
        "file": "dickens_oliver_twist.txt",
        "doc_id": "dickens_oliver_twist",
        "title": "Oliver Twist",
        "author": "Charles Dickens",
        "role": "train",
    },
    {
        "file": "dickens_two_cities.txt",
        "doc_id": "dickens_two_cities",
        "title": "A Tale of Two Cities",
        "author": "Charles Dickens",
        "role": "test",
    },
    {
        "file": "doyle_sherlock.txt",
        "doc_id": "doyle_sherlock",
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "role": "train",
    },
    {
        "file": "doyle_baskervilles.txt",
        "doc_id": "doyle_baskervilles",
        "title": "The Hound of the Baskervilles",
        "author": "Arthur Conan Doyle",
        "role": "train",
    },
    {
        "file": "doyle_lost_world.txt",
        "doc_id": "doyle_lost_world",
        "title": "The Lost World",
        "author": "Arthur Conan Doyle",
        "role": "test",
    },
    {
        "file": "wells_time_machine.txt",
        "doc_id": "wells_time_machine",
        "title": "The Time Machine",
        "author": "H. G. Wells",
        "role": "train",
    },
    {
        "file": "wells_war_of_worlds.txt",
        "doc_id": "wells_war_of_worlds",
        "title": "The War of the Worlds",
        "author": "H. G. Wells",
        "role": "train",
    },
    {
        "file": "wells_invisible_man.txt",
        "doc_id": "wells_invisible_man",
        "title": "The Invisible Man",
        "author": "H. G. Wells",
        "role": "test",
    },
]


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Remove Project Gutenberg header and footer license blocks."""
    start_match = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", raw_text)
    end_match = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", raw_text)
    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)
    body = raw_text[start_idx:end_idx].strip()
    # Skip initial front-matter table of contents (take middle 85% of novel prose to avoid title/author byline)
    n = len(body)
    return body[int(n * 0.03) : int(n * 0.97)].strip()


def load_literary_novelists_corpus(
    novels_dir: str | Path,
    segment_words: Optional[int] = 10000,
    max_segments_per_novel: int = 3,
) -> List[CorpusDocument]:
    """
    Load the 15 classic novels by Austen, C. Brontë, Dickens, Doyle, and Wells.
    If `segment_words` is provided, extracts up to `max_segments_per_novel` non-overlapping
    blocks of `segment_words` words per novel (producing 45 balanced 10,000-word literary
    documents across the 5 authors), or if `segment_words=None`, returns the 15 full novels.
    """
    base = Path(novels_dir)
    docs: List[CorpusDocument] = []

    for meta in NOVEL_METADATA:
        fpath = base / meta["file"]
        if not fpath.exists():
            continue
        clean_text = strip_gutenberg_boilerplate(fpath.read_text(encoding="utf-8", errors="ignore"))

        if segment_words is None:
            words = tokenize_words(clean_text)
            docs.append(
                CorpusDocument(
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    author=meta["author"],
                    status=meta["role"],
                    text=clean_text,
                    word_count=len(words),
                    metadata={"role": meta["role"], "novel": meta["title"]},
                )
            )
        else:
            # Split into equal-word-count raw text segments by whitespace tokens
            raw_tokens = clean_text.split()
            for seg_idx in range(max_segments_per_novel):
                start = seg_idx * segment_words
                end = start + segment_words
                if end > len(raw_tokens):
                    break
                seg_text = " ".join(raw_tokens[start:end])
                words = tokenize_words(seg_text)
                docs.append(
                    CorpusDocument(
                        doc_id=f"{meta['doc_id']}_part{seg_idx + 1}",
                        title=f"{meta['title']} (Part {seg_idx + 1})",
                        author=meta["author"],
                        status=meta["role"],
                        text=seg_text,
                        word_count=len(words),
                        metadata={
                            "role": meta["role"],
                            "novel": meta["title"],
                            "novel_id": meta["doc_id"],
                            "segment": seg_idx + 1,
                        },
                    )
                )

    return docs
