"""
Registry of stylometric analysis resources, canonical word lists,
method specifications, software tools, benchmark corpora, and references.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# 1. Canonical Function Word Lists for Authorship Attribution
# ---------------------------------------------------------------------------

# Mosteller & Wallace (1964) high-discriminant marker function words used to
# resolve the disputed Federalist Papers between Hamilton and Madison.
# Group 1: High-discriminant words (including 'upon', 'whilst', 'while', 'enough',
# 'there', 'on', 'to', 'by', 'of', 'from', 'in', 'also', 'an', 'both', 'though',
# 'although', 'commonly', 'consequently', 'considerable', 'according', 'apt',
# 'direction', 'innovation', 'language', 'vigor', 'kind', 'matter', 'particularly',
# 'probability', 'work').
MOSTELLER_WALLACE_MARKERS: List[str] = [
    "upon",
    "whilst",
    "while",
    "enough",
    "there",
    "on",
    "to",
    "by",
    "of",
    "from",
    "in",
    "also",
    "an",
    "both",
    "though",
    "although",
    "commonly",
    "consequently",
    "considerable",
    "according",
    "apt",
    "direction",
    "innovation",
    "language",
    "vigor",
    "kind",
    "matter",
    "particularly",
    "probability",
    "work",
]

# Contrastive pairs highlighted in Mosteller & Wallace (1964):
# Hamilton strongly favors 'upon' over 'on', 'while' over 'whilst', and 'enough'.
# Madison strongly favors 'whilst' over 'while', 'on' over 'upon', and 'consequently'.
MOSTELLER_WALLACE_CONTRASTIVE_PAIRS: Dict[str, Dict[str, str]] = {
    "upon_vs_on": {
        "hamilton_marker": "upon",
        "madison_marker": "on",
        "note": "Hamilton uses 'upon' ~3.2 per 1,000 words vs. Madison's ~0.16 per 1,000 words.",
    },
    "while_vs_whilst": {
        "hamilton_marker": "while",
        "madison_marker": "whilst",
        "note": "Hamilton almost exclusively uses 'while'; Madison almost exclusively uses 'whilst'.",
    },
    "enough": {
        "hamilton_marker": "enough",
        "madison_marker": "",
        "note": "Hamilton uses 'enough' regularly; Madison virtually never uses it in the Federalist.",
    },
    "consequently": {
        "hamilton_marker": "",
        "madison_marker": "consequently",
        "note": "Madison uses 'consequently' at a much higher rate than Hamilton.",
    },
}

# Standard English Function Words (180 topic-neutral closed-class words:
# determiners, prepositions, conjunctions, auxiliary/modal verbs, pronouns,
# quantifiers, negations, and discourse particles).
STANDARD_FUNCTION_WORDS: List[str] = [
    # Articles & Determiners
    "the", "a", "an", "this", "that", "these", "those", "such", "another",
    "each", "every", "either", "neither", "both", "all", "any", "some", "no",
    "none", "few", "many", "much", "more", "most", "other", "several", "enough",
    "half", "less", "least", "own", "same", "very",
    # Prepositions
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "amongst",
    "upon", "unto", "within", "towards", "toward", "throughout", "despite",
    "beyond", "across", "behind", "Beneath", "beside", "besides", "near",
    "off", "above", "below", "up", "down", "since", "until", "till", "concerning",
    "regarding", "respecting",
    # Conjunctions & Subordinators
    "and", "but", "or", "nor", "for", "yet", "so", "although", "though",
    "even", "because", "unless", "whereas", "while", "whilst", "where",
    "whereby", "wherein", "whereof", "whether", "if", "than", "that", "lest",
    "once", "Provided", "inasmuch", "insomuch",
    # Pronouns & Relative Words
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "one", "oneself",
    "who", "whom", "whose", "which", "what", "whatever", "whoever", "whichever",
    # Auxiliary & Modal Verbs
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "done", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "ought", "dare", "need",
    # Adverbs, Negations & Discourse Connectors
    "not", "no", "never", "neither", "nor", "here", "there", "now", "then",
    "when", "always", "often", "sometimes", "seldom", "ever", "already",
    "still", "just", "only", "also", "too", "quite", "rather", "almost",
    "nearly", "scarcely", "hardly", "merely", "simply", "indeed", "perhaps",
    "therefore", "thus", "hence", "consequently", "accordingly", "however",
    "nevertheless", "nonetheless", "moreover", "furthermore", "otherwise",
    "instead", "meanwhile",
]
# Deduplicate & lowercase while preserving order
STANDARD_FUNCTION_WORDS = list(dict.fromkeys(w.lower() for w in STANDARD_FUNCTION_WORDS))

PERSONAL_PRONOUNS: Set[str] = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
}


# ---------------------------------------------------------------------------
# 2. Catalog of Stylometric Methods, Software Libraries, and Corpora
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StylometricMethodResource:
    id: str
    name: str
    category: str
    year: int
    authors: str
    formula_summary: str
    description: str
    implemented_in_module: str


@dataclass(frozen=True)
class ExternalSoftwareResource:
    name: str
    language: str
    authors_or_org: str
    url: str
    key_capabilities: str


@dataclass(frozen=True)
class BenchmarkCorpusResource:
    id: str
    name: str
    authors_included: List[str]
    document_count: int
    historical_problem: str
    source_url: str


STYLOMETRIC_METHODS_CATALOG: List[StylometricMethodResource] = [
    StylometricMethodResource(
        id="mendenhall_curves",
        name="Mendenhall's Characteristic Curves of Composition",
        category="Lexical / Word-Length Distribution",
        year=1887,
        authors="T. C. Mendenhall",
        formula_summary="P(L = k) = count(|w| = k) / N for k in 1..max_len",
        description=(
            "Pioneering stylometric fingerprint measuring the relative frequency "
            "distribution of word lengths (1-letter, 2-letter, ..., k-letter words). "
            "Applied by Mendenhall (1887, 1901) to Shakespeare, Marlowe, and Bacon."
        ),
        implemented_in_module="stylometry.lexical.MendenhallCurveAnalyzer",
    ),
    StylometricMethodResource(
        id="lexical_diversity_suite",
        name="Lexical Richness & Vocabulary Diversity Indices",
        category="Lexical Richness / Type-Token Statistics",
        year=1944,
        authors="G. U. Yule (1944), E. H. Simpson (1949), P. Guiraud (1954), A. Honoré (1979), M. A. Covington & J. D. McFall (2010)",
        formula_summary="TTR = V/N; Guiraud R = V/sqrt(N); Yule's K = 10^4 * (sum(i^2 * V_i) - N) / N^2; MATTR(W)",
        description=(
            "Quantifies an author's vocabulary richness and repetition rate independent "
            "of text length using Yule's K Characteristic, Simpson's D, Honoré's R "
            "(hapax legomena scaling), Sichel's S (dislegomena ratio), Brunét's W, "
            "and Moving-Average Type-Token Ratio (MATTR)."
        ),
        implemented_in_module="stylometry.lexical.compute_lexical_metrics",
    ),
    StylometricMethodResource(
        id="mosteller_wallace_discriminant",
        name="Mosteller & Wallace Function-Word Discriminant Analysis",
        category="Function-Word Probabilistic / Discriminant Attribution",
        year=1964,
        authors="Frederick Mosteller & David L. Wallace",
        formula_summary="Log-odds / Bayesian & Linear Discriminant weighting on topic-independent marker words ('upon', 'whilst', 'enough', etc.)",
        description=(
            "Landmark statistical study resolving the 12 disputed Federalist Papers "
            "by modeling rates of non-contextual function words (prepositions, "
            "conjunctions, adverbs) whose usage is invariant to political topic."
        ),
        implemented_in_module="stylometry.classifiers.MostellerWallaceClassifier",
    ),
    StylometricMethodResource(
        id="kilgarriff_chi_squared",
        name="Kilgarriff's Chi-Squared (χ²) Distance",
        category="Statistical Frequency Distance",
        year=2001,
        authors="Adam Kilgarriff",
        formula_summary="chi^2(A, B) = sum_{w in MFW} (O_{w,A} - E_{w,A})^2 / E_{w,A} + (O_{w,B} - E_{w,B})^2 / E_{w,B}",
        description=(
            "Measures corpus similarity and authorship by computing the Pearson "
            "chi-squared statistic over raw token counts of the top-N most frequent "
            "words between a candidate author's subcorpus and an unknown text."
        ),
        implemented_in_module="stylometry.chi_square.KilgarriffChiSquare",
    ),
    StylometricMethodResource(
        id="burrows_delta",
        name="Burrows's Classic Delta (Manhattan Z-Score Distance)",
        category="Delta Family (MFW Z-Score Distances)",
        year=2002,
        authors="John F. Burrows",
        formula_summary="Delta_B(D_1, D_2) = (1/M) * sum_{i=1}^M |z_i(D_1) - z_i(D_2)|",
        description=(
            "The foundational modern stylometric metric: standardizes relative "
            "frequencies of the M Most Frequent Words (MFWs) into corpus z-scores "
            "so all features have equal variance, then computes mean absolute "
            "(L1 / Manhattan) distance."
        ),
        implemented_in_module="stylometry.delta.DeltaAnalyzer (metric='burrows')",
    ),
    StylometricMethodResource(
        id="argamon_quadratic_delta",
        name="Argamon's Quadratic Delta (Euclidean Z-Score Distance)",
        category="Delta Family (MFW Z-Score Distances)",
        year=2008,
        authors="Shlomo Argamon",
        formula_summary="Delta_Q(D_1, D_2) = sqrt( (1/M) * sum_{i=1}^M (z_i(D_1) - z_i(D_2))^2 )",
        description=(
            "Probabilistic derivation of Burrows's Delta under Gaussian independence "
            "assumptions, replacing the L1 norm on z-scores with the L2 (Euclidean) distance."
        ),
        implemented_in_module="stylometry.delta.DeltaAnalyzer (metric='quadratic')",
    ),
    StylometricMethodResource(
        id="cosine_wurzburg_delta",
        name="Cosine Delta (Würzburg Delta)",
        category="Delta Family (MFW Z-Score Distances)",
        year=2017,
        authors="Peter W. H. Smith & W. Aldridge (2011); Stefan Evert, Thomas Proisl, Fotis Jannidis et al. (2017)",
        formula_summary="Delta_cos(D_1, D_2) = 1 - (z(D_1) . z(D_2)) / (||z(D_1)||_2 * ||z(D_2)||_2)",
        description=(
            "Normalizes z-score vectors to the unit hypersphere before computing angular "
            "(cosine) distance. Demonstrated by Evert et al. (2017) to outperform classic "
            "Burrows's Delta across languages, text lengths, and large MFW sizes."
        ),
        implemented_in_module="stylometry.delta.DeltaAnalyzer (metric='cosine')",
    ),
    StylometricMethodResource(
        id="eder_delta",
        name="Eder's Delta (Rank-Weighted Manhattan Z-Score Distance)",
        category="Delta Family (MFW Z-Score Distances)",
        year=2016,
        authors="Maciej Eder, Jan Rybicki & Mike Kestemont",
        formula_summary="Delta_E(D_1, D_2) = sum_{i=1}^M ((M - i + 1) / M) * |z_i(D_1) - z_i(D_2)|",
        description=(
            "Weights each feature in the MFW list by its inverse frequency rank so that "
            "higher-frequency grammatical words exert stronger influence than lower-ranked words."
        ),
        implemented_in_module="stylometry.delta.DeltaAnalyzer (metric='eder')",
    ),
    StylometricMethodResource(
        id="char_ngram_profiles",
        name="Character N-Gram Profile Attribution (CNG)",
        category="Sub-word / Morphological Stylometry",
        year=2003,
        authors="Vlado Kešelj et al. (2003); Efstathios Stamatatos (2009)",
        formula_summary="Cosine / Delta distance over top-K character 3-grams and 4-grams",
        description=(
            "Captures sub-word prefixes, suffixes, inflectional morphemes, word boundaries, "
            "and punctuation-whitespace habits without requiring language-specific lemmatizers."
        ),
        implemented_in_module="stylometry.syntactic.CharNGramAnalyzer",
    ),
    StylometricMethodResource(
        id="syntactic_punctuation_fingerprint",
        name="Syntactic Complexity, Sentence Rhythm & Punctuation Fingerprinting",
        category="Syntactic & Structural Stylometry",
        year=1998,
        authors="E. Stamatatos et al. (2000); H. Baayen et al. (1996)",
        formula_summary="Sentence length distribution moments + punctuation rates per 1,000 tokens (; : , — ? ! ( )) + readability indices",
        description=(
            "Measures clause hierarchy and authorial cadence via sentence-length distributions, "
            "semicolon/colon/dash habits, and readability complexity metrics."
        ),
        implemented_in_module="stylometry.syntactic.compute_syntactic_metrics",
    ),
    StylometricMethodResource(
        id="koppel_general_imposters",
        name="Koppel & Winter's General Imposters (GI) Authorship Verification",
        category="Open-Set Authorship Verification",
        year=2014,
        authors="Moshe Koppel & Yaron Winter (2014); Mike Kestemont et al. (2016)",
        formula_summary="S(X, C) = (1/K) * sum_{k=1}^K I[ min_{c in C} d_{F_k}(X, c) < min_{imp in I_k} d_{F_k}(X, imp) ]",
        description=(
            "Gold-standard open-set authorship verification algorithm: across K Monte Carlo "
            "iterations, randomly samples 50% of features and a subset of distractor ('imposter') "
            "documents, measuring the proportion of iterations where the disputed text is "
            "closer to the candidate author than to all sampled imposters."
        ),
        implemented_in_module="stylometry.imposters.GeneralImpostersVerifier",
    ),
    StylometricMethodResource(
        id="multivariate_ml_attribution",
        name="Multivariate & Supervised Machine Learning Attribution (SVM, Logistic Regression, PCA, Ward Clustering)",
        category="Multivariate & Supervised Machine Learning",
        year=2009,
        authors="J. F. Burrows (1987 PCA); G. Fung (2003 SVM); E. Stamatatos (2009)",
        formula_summary="Linear SVM / L2 Logistic Regression / PCA + Ward Hierarchical Clustering on composite stylometric feature matrices",
        description=(
            "Combines function-word frequencies, lexical diversity, sentence rhythm, "
            "punctuation density, and character n-grams into calibrated supervised classifiers "
            "and unsupervised PCA/dendrogram projections."
        ),
        implemented_in_module="stylometry.classifiers.StylometricEnsembleAttributor",
    ),
    StylometricMethodResource(
        id="craig_burrows_zeta",
        name="Burrows's & Craig's Zeta (ζ) Distinctive Vocabulary Analysis",
        category="Distinctive Vocabulary / Mid-Frequency Marker Analysis",
        year=2007,
        authors="John F. Burrows (2007); Hugh Craig & Arthur F. Kinney (2009)",
        formula_summary="zeta(w) = P(w in S_A) - P(w in S_B) across fixed-size text segments",
        description=(
            "Identifies mid-frequency lexical words consistently used across one author's "
            "text segments while consistently avoided by another author."
        ),
        implemented_in_module="stylometry.zeta.ZetaAnalyzer",
    ),
    StylometricMethodResource(
        id="koppel_unmasking",
        name="Koppel & Schler's 'Unmasking' Degradation Curves",
        category="Open-Set Authorship Verification",
        year=2004,
        authors="Moshe Koppel & Jonathan Schler",
        formula_summary="Iterative linear SVM/LogReg cross-validation accuracy degradation after removing top-2k separating features per round",
        description=(
            "Measures the 'depth of difference' between two texts: same-author texts "
            "experience a steep accuracy drop when a handful of topical markers are "
            "removed, whereas different-author texts remain separable across iterations."
        ),
        implemented_in_module="stylometry.zeta.UnmaskingVerifier",
    ),
    StylometricMethodResource(
        id="eder_rolling_stylometry",
        name="Rolling Stylometry (Sequential Collaborative Authorship Analysis)",
        category="Sequential / Collaborative Authorship Analysis",
        year=2016,
        authors="Maciej Eder",
        formula_summary="Sliding-window Delta / SVM trajectory d(W_t, Author_c) over windows W_t of length L and step S",
        description=(
            "Detects co-authorship, interpolations, and stylistic transitions within a "
            "single work by evaluating stylometric signatures across overlapping windows."
        ),
        implemented_in_module="stylometry.rolling.RollingStylometryAnalyzer",
    ),
    StylometricMethodResource(
        id="eder_bootstrap_consensus",
        name="Bootstrap Consensus Tree (BCT) & Multi-MFW Stability Analysis",
        category="Consensus & Parameter Stability",
        year=2012,
        authors="Maciej Eder (2012, 2016); Jan Rybicki",
        formula_summary="Consensus vote across MFW grid M in {50..500} x culling ratios x Delta metrics",
        description=(
            "Eliminates sensitivity to a single arbitrary MFW cutoff by aggregating "
            "attributions across a multi-parameter grid of MFW sizes and culling thresholds."
        ),
        implemented_in_module="stylometry.consensus_tree.BootstrapConsensusAnalyzer",
    ),
    StylometricMethodResource(
        id="pos_ngram_stylometry",
        name="Morphosyntactic & POS Tag N-Gram Transition Stylometry",
        category="Syntactic Sequence Stylometry",
        year=2007,
        authors="Graeme Hirst & Ol'ga Feiguina (2007); Efstathios Stamatatos (2009)",
        formula_summary="Cosine Delta over standardized morphosyntactic / POS tag 1-grams, 2-grams, and 3-grams",
        description=(
            "Captures an author's unconscious syntactic phrase templates and grammatical "
            "transition preferences independent of surface vocabulary."
        ),
        implemented_in_module="stylometry.pos_stylometry.POSTagNGramAnalyzer",
    ),
]


EXTERNAL_SOFTWARE_RESOURCES: List[ExternalSoftwareResource] = [
    ExternalSoftwareResource(
        name="stylo (R Package)",
        language="R",
        authors_or_org="Computational Stylistics Group (Maciej Eder, Jan Rybicki, Mike Kestemont)",
        url="https://github.com/computationalstylistics/stylo",
        key_capabilities="De facto standard in Digital Humanities for Delta (Burrows, Cosine, Eder), PCA, Bootstrap Consensus Trees, Rolling Stylometry, and General Imposters (`imposters()`).",
    ),
    ExternalSoftwareResource(
        name="faststylometry",
        language="Python",
        authors_or_org="Fast Data Science (Thomas Wood)",
        url="https://github.com/fastdatascience/faststylometry",
        key_capabilities="Burrows's Delta implementation with logistic-calibrated probability estimation for same-author vs. different-author pairs.",
    ),
    ExternalSoftwareResource(
        name="StyloMetrix",
        language="Python",
        authors_or_org="ZILiAT-NASK",
        url="https://github.com/ZILiAT-NASK/StyloMetrix",
        key_capabilities="Interpretable grammatical, syntactic, and lexical feature vectors built on spaCy.",
    ),
    ExternalSoftwareResource(
        name="pystyl",
        language="Python",
        authors_or_org="Mike Kestemont et al.",
        url="https://github.com/mikekestemont/pystyl",
        key_capabilities="Python stylometry preprocessing, MFW vectorization, distance matrices, and clustering.",
    ),
    ExternalSoftwareResource(
        name="Programming Historian Stylometry Suite",
        language="Python / NLTK",
        authors_or_org="François Dominic Laramée",
        url="https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python",
        key_capabilities="Canonical Python implementations of Mendenhall's Characteristic Curves, Kilgarriff's Chi-Squared, and Burrows's Delta on The Federalist Papers.",
    ),
    ExternalSoftwareResource(
        name="PAN Clef Shared Tasks on Digital Text Forensics and Stylometry",
        language="Multi-language / Benchmarks",
        authors_or_org="PAN Workshop Series (Webis Group)",
        url="https://pan.webis.de/",
        key_capabilities="Annual benchmark datasets and evaluation harnesses for authorship attribution, authorship verification, style change detection, and AI text detection.",
    ),
]


BENCHMARK_CORPORA_CATALOG: List[BenchmarkCorpusResource] = [
    BenchmarkCorpusResource(
        id="federalist_papers",
        name="The Federalist Papers (1787–1788)",
        authors_included=["Alexander Hamilton", "James Madison", "John Jay"],
        document_count=85,
        historical_problem=(
            "85 essays published under the shared pseudonym 'Publius'. 51 undisputed by Hamilton, "
            "14 undisputed by Madison, 5 by Jay (#2-5, #64), 3 joint (#18-20), and 12 famous "
            "disputed essays (#49-58, #62-63) claimed by both Hamilton and Madison."
        ),
        source_url="https://www.gutenberg.org/ebooks/1404",
    ),
    BenchmarkCorpusResource(
        id="classic_british_novelists",
        name="19th-Century Classic Literature Benchmark Corpus",
        authors_included=[
            "Jane Austen",
            "Charlotte Brontë",
            "Charles Dickens",
            "Sir Arthur Conan Doyle",
            "H. G. Wells",
        ],
        document_count=15,
        historical_problem=(
            "Multi-author literary attribution across 15 full novels (~1.9M words), including "
            "cross-genre test cases such as attributing Sir Arthur Conan Doyle's science-fiction "
            "adventure 'The Lost World' against H. G. Wells's science-fiction romances and "
            "distinguishing Victorian contemporaries."
        ),
        source_url="https://www.gutenberg.org/",
    ),
]


def get_all_resources_summary() -> Dict[str, object]:
    """Return a structured dictionary of all stylometric resources in the package."""
    return {
        "methods": [asdict(m) for m in STYLOMETRIC_METHODS_CATALOG],
        "external_software": [asdict(s) for s in EXTERNAL_SOFTWARE_RESOURCES],
        "benchmark_corpora": [asdict(c) for c in BENCHMARK_CORPORA_CATALOG],
        "word_lists": {
            "mosteller_wallace_markers_count": len(MOSTELLER_WALLACE_MARKERS),
            "mosteller_wallace_markers": MOSTELLER_WALLACE_MARKERS,
            "mosteller_wallace_contrastive_pairs": MOSTELLER_WALLACE_CONTRASTIVE_PAIRS,
            "standard_function_words_count": len(STANDARD_FUNCTION_WORDS),
            "personal_pronouns_count": len(PERSONAL_PRONOUNS),
        },
    }
