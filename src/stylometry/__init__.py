"""
`stylometry` — Unified Computational Stylometry & Authorship Attribution Package.

Integrates classical and modern stylometric methods:
- Lexical Richness & Vocabulary Diversity (TTR, Guiraud's R, Herdan's C, MATTR,
  Yule's K, Simpson's D, Honoré's R, Sichel's S, Brunét's W)
- Mendenhall's (1887) Characteristic Curves of Composition
- Mosteller & Wallace (1964) Function-Word Discriminant Analysis
- Kilgarriff's (2001) Chi-Squared (χ²) Distance
- The Delta Family:
    * Burrows's Classic Delta (2002)
    * Argamon's Quadratic Delta (2008)
    * Cosine / Würzburg Delta (Smith & Aldridge 2011; Evert et al. 2017)
    * Eder's Rank-Weighted Delta (2016)
    * Canberra & MinMax (Ruzicka) Distances
- Character N-Gram Profile Attribution (Kešelj et al. 2003; Stamatatos 2009)
- Morphosyntactic & POS Tag N-Gram Transition Stylometry (Hirst & Feiguina 2007)
- Syntactic Rhythm, Punctuation Fingerprinting & Readability Metrics
- Burrows's & Craig's Zeta (ζ) Distinctive Vocabulary Analysis (2007, 2009)
- Koppel & Schler's (2004) 'Unmasking' Degradation Curves
- Koppel & Winter's (2014) General Imposters (GI) Authorship Verification
- Eder's (2016) Rolling Stylometry for Sequential / Collaborative Authorship
- Eder's (2012, 2016) Bootstrap Consensus Trees (BCT) & Multi-MFW Stability
- Multivariate Supervised ML (Linear SVM, L2 Logistic Regression), PCA, Ward
  Hierarchical Clustering, and 12-Method Consensus Ensemble Attribution.
"""

from stylometry.resources import (
    BENCHMARK_CORPORA_CATALOG,
    EXTERNAL_SOFTWARE_RESOURCES,
    MOSTELLER_WALLACE_CONTRASTIVE_PAIRS,
    MOSTELLER_WALLACE_MARKERS,
    PERSONAL_PRONOUNS,
    STANDARD_FUNCTION_WORDS,
    STYLOMETRIC_METHODS_CATALOG,
    get_all_resources_summary,
)
from stylometry.preprocessing import (
    chunk_tokens,
    normalize_text,
    select_mfw,
    tokenize_sentences,
    tokenize_words,
)
from stylometry.lexical import (
    MendenhallCurveAnalyzer,
    compute_lexical_metrics,
    compute_mattr,
)
from stylometry.syntactic import (
    CharNGramAnalyzer,
    compute_function_word_rates,
    compute_syntactic_metrics,
    count_syllables,
)
from stylometry.delta import (
    DeltaAnalyzer,
    compute_delta_vector_distance,
)
from stylometry.chi_square import KilgarriffChiSquare
from stylometry.imposters import GeneralImpostersVerifier
from stylometry.zeta import (
    UnmaskingVerifier,
    ZetaAnalyzer,
)
from stylometry.rolling import RollingStylometryAnalyzer
from stylometry.consensus_tree import BootstrapConsensusAnalyzer
from stylometry.pos_stylometry import (
    POSTagNGramAnalyzer,
    extract_pos_tag_sequence,
    morphosyntactic_tag,
)
from stylometry.classifiers import (
    CompositeStylometricClassifier,
    MostellerWallaceClassifier,
    StylometricEnsembleAttributor,
    compute_pca_and_clustering,
    evaluate_methods_cross_validation,
)
from stylometry.corpus import (
    CorpusDocument,
    DISPUTED_PAPERS,
    HAMILTON_PAPERS,
    JAY_PAPERS,
    JOINT_PAPERS,
    MADISON_PAPERS,
    load_literary_novelists_corpus,
    parse_federalist_corpus,
)

__version__ = "1.1.0"

__all__ = [
    "BENCHMARK_CORPORA_CATALOG",
    "EXTERNAL_SOFTWARE_RESOURCES",
    "MOSTELLER_WALLACE_CONTRASTIVE_PAIRS",
    "MOSTELLER_WALLACE_MARKERS",
    "PERSONAL_PRONOUNS",
    "STANDARD_FUNCTION_WORDS",
    "STYLOMETRIC_METHODS_CATALOG",
    "get_all_resources_summary",
    "chunk_tokens",
    "normalize_text",
    "select_mfw",
    "tokenize_sentences",
    "tokenize_words",
    "MendenhallCurveAnalyzer",
    "compute_lexical_metrics",
    "compute_mattr",
    "CharNGramAnalyzer",
    "compute_function_word_rates",
    "compute_syntactic_metrics",
    "count_syllables",
    "DeltaAnalyzer",
    "compute_delta_vector_distance",
    "KilgarriffChiSquare",
    "GeneralImpostersVerifier",
    "UnmaskingVerifier",
    "ZetaAnalyzer",
    "RollingStylometryAnalyzer",
    "BootstrapConsensusAnalyzer",
    "POSTagNGramAnalyzer",
    "extract_pos_tag_sequence",
    "morphosyntactic_tag",
    "CompositeStylometricClassifier",
    "MostellerWallaceClassifier",
    "StylometricEnsembleAttributor",
    "compute_pca_and_clustering",
    "evaluate_methods_cross_validation",
    "CorpusDocument",
    "DISPUTED_PAPERS",
    "HAMILTON_PAPERS",
    "JAY_PAPERS",
    "JOINT_PAPERS",
    "MADISON_PAPERS",
    "load_literary_novelists_corpus",
    "parse_federalist_corpus",
]
