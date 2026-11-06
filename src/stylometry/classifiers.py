"""
Multivariate stylometric classifiers, Mosteller & Wallace (1964) discriminant
analysis, PCA / hierarchical clustering, and multi-method consensus attribution.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from stylometry.chi_square import KilgarriffChiSquare
from stylometry.delta import DeltaAnalyzer
from stylometry.imposters import GeneralImpostersVerifier
from stylometry.lexical import MendenhallCurveAnalyzer, compute_lexical_metrics
from stylometry.resources import MOSTELLER_WALLACE_MARKERS, STANDARD_FUNCTION_WORDS
from stylometry.syntactic import (
    CharNGramAnalyzer,
    compute_function_word_rates,
    compute_syntactic_metrics,
)


class MostellerWallaceClassifier:
    """
    Implements a modern regularized discriminant & Poisson/Log-Odds classifier
    over the 30 high-discriminant marker words identified by Mosteller & Wallace
    (1964) in 'Inference and Disputed Authorship: The Federalist'.

    Provides both class probability predictions and per-word log-likelihood /
    weight breakdowns (e.g., 'upon', 'whilst', 'enough', 'while', 'on', 'to').
    """

    def __init__(
        self,
        marker_words: Optional[Sequence[str]] = None,
        C: float = 1.0,
    ):
        self.marker_words = list(marker_words or MOSTELLER_WALLACE_MARKERS)
        self.C = C
        self.scaler_ = StandardScaler()
        self.clf_ = LogisticRegression(C=self.C, max_iter=2000, random_state=42)
        self.classes_: List[str] = []
        self.author_mean_rates_: Dict[str, Dict[str, float]] = {}

    def _featurize_one(self, text: str) -> np.ndarray:
        rates = compute_function_word_rates(
            text, word_list=self.marker_words, per_n_tokens=1000.0
        )
        return np.array([rates[w] for w in self.marker_words], dtype=float)

    def _featurize(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._featurize_one(t) for t in texts])

    def fit(self, texts: Sequence[str], authors: Sequence[str]) -> "MostellerWallaceClassifier":
        X = self._featurize(texts)
        X_scaled = self.scaler_.fit_transform(X)
        self.clf_.fit(X_scaled, authors)
        self.classes_ = list(self.clf_.classes_)

        self.author_mean_rates_ = {}
        for a in sorted(set(authors)):
            idx = [i for i, doc_a in enumerate(authors) if doc_a == a]
            mean_vec = np.mean(X[idx], axis=0)
            self.author_mean_rates_[a] = {
                w: round(float(mean_vec[j]), 3) for j, w in enumerate(self.marker_words)
            }
        return self

    def predict_one(self, text: str) -> Tuple[str, Dict[str, float]]:
        x = self._featurize_one(text).reshape(1, -1)
        x_scaled = self.scaler_.transform(x)
        probs = self.clf_.predict_proba(x_scaled)[0]
        prob_dict = {cls: float(p) for cls, p in zip(self.classes_, probs)}
        best = max(prob_dict, key=prob_dict.get)
        return best, prob_dict

    def predict(self, texts: Sequence[str]) -> List[str]:
        X = self._featurize(texts)
        X_scaled = self.scaler_.transform(X)
        return [str(p) for p in self.clf_.predict(X_scaled)]


class CompositeStylometricClassifier:
    """
    Supervised machine learning classifier (Linear SVM or L2 Logistic Regression)
    operating on a composite stylometric feature vector that fuses:
      1. Function word rates (180 topic-independent function words per 1,000 tokens)
      2. Lexical diversity metrics (MATTR, Yule's K, Guiraud's R, Honoré's R, Hapax/Dislegomena)
      3. Syntactic & punctuation rhythm (sentence length statistics, semicolon/colon/dash/comma
         rates per 1,000 tokens, and readability indices).
    """

    LEXICAL_KEYS = [
        "mean_word_length",
        "std_word_length",
        "root_ttr_guiraud",
        "log_ttr_herdan",
        "mattr",
        "hapax_ratio",
        "dislegomena_ratio",
        "yules_k",
        "simpsons_d",
        "honores_r",
    ]

    SYNTACTIC_KEYS = [
        "mean_sentence_length",
        "std_sentence_length",
        "median_sentence_length",
        "iqr_sentence_length",
        "short_sentence_ratio",
        "long_sentence_ratio",
        "comma_rate_per_1k",
        "semicolon_rate_per_1k",
        "colon_rate_per_1k",
        "dash_rate_per_1k",
        "question_rate_per_1k",
        "exclamation_rate_per_1k",
        "paren_rate_per_1k",
        "mean_syllables_per_word",
        "complex_word_ratio",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "coleman_liau_index",
        "gunning_fog_index",
    ]

    def __init__(self, model_type: str = "logreg", C: float = 0.5):
        self.model_type = model_type
        self.C = C
        self.feature_names_: List[str] = (
            [f"fw_{w}" for w in STANDARD_FUNCTION_WORDS]
            + [f"lex_{k}" for k in self.LEXICAL_KEYS]
            + [f"syn_{k}" for k in self.SYNTACTIC_KEYS]
        )
        self.scaler_ = StandardScaler()
        if model_type == "svm":
            base_svm = LinearSVC(C=self.C, max_iter=5000, random_state=42)
            self.clf_ = CalibratedClassifierCV(base_svm, cv=3)
        else:
            self.clf_ = LogisticRegression(C=self.C, max_iter=3000, random_state=42)
        self.classes_: List[str] = []

    def extract_vector(self, text: str) -> np.ndarray:
        fw = compute_function_word_rates(text, word_list=STANDARD_FUNCTION_WORDS)
        lex = compute_lexical_metrics(text)
        syn = compute_syntactic_metrics(text)

        vec = (
            [fw[w] for w in STANDARD_FUNCTION_WORDS]
            + [lex[k] for k in self.LEXICAL_KEYS]
            + [syn[k] for k in self.SYNTACTIC_KEYS]
        )
        return np.array(vec, dtype=float)

    def fit(self, texts: Sequence[str], authors: Sequence[str]) -> "CompositeStylometricClassifier":
        X = np.vstack([self.extract_vector(t) for t in texts])
        X_scaled = self.scaler_.fit_transform(X)
        self.clf_.fit(X_scaled, authors)
        self.classes_ = list(self.clf_.classes_)
        return self

    def predict_one(self, text: str) -> Tuple[str, Dict[str, float]]:
        x = self.extract_vector(text).reshape(1, -1)
        x_scaled = self.scaler_.transform(x)
        probs = self.clf_.predict_proba(x_scaled)[0]
        prob_dict = {cls: float(p) for cls, p in zip(self.classes_, probs)}
        best = max(prob_dict, key=prob_dict.get)
        return best, prob_dict

    def predict(self, texts: Sequence[str]) -> List[str]:
        X = np.vstack([self.extract_vector(t) for t in texts])
        X_scaled = self.scaler_.transform(X)
        return [str(p) for p in self.clf_.predict(X_scaled)]


def compute_pca_and_clustering(
    texts: Sequence[str],
    labels: Sequence[str],
    doc_ids: Sequence[str],
    n_mfw: int = 150,
) -> Dict[str, object]:
    """
    Perform unsupervised Principal Component Analysis (PCA) and Ward's
    Hierarchical Agglomerative Clustering on the standardized MFW z-score matrix.
    Returns 2D/3D coordinates, explained variance ratios, top feature loadings,
    and linkage tree steps.
    """
    delta = DeltaAnalyzer(n_mfw=n_mfw, metric="cosine", profile_mode="document_centroid")
    delta.fit(texts, labels)
    z_mat = delta.doc_z_matrix_

    n_comp = min(3, z_mat.shape[0], z_mat.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(z_mat)

    # Top loadings for PC1 and PC2
    pc1_loadings = sorted(
        zip(delta.mfw_, pca.components_[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:15]
    pc2_loadings = (
        sorted(
            zip(delta.mfw_, pca.components_[1]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:15]
        if n_comp >= 2
        else []
    )

    points = []
    for i in range(len(texts)):
        points.append(
            {
                "doc_id": doc_ids[i],
                "label": labels[i],
                "pc1": round(float(coords[i, 0]), 4),
                "pc2": round(float(coords[i, 1]), 4) if n_comp >= 2 else 0.0,
                "pc3": round(float(coords[i, 2]), 4) if n_comp >= 3 else 0.0,
            }
        )

    # Ward hierarchical clustering on Euclidean z-distances
    dist_condensed = pdist(z_mat, metric="euclidean")
    Z_link = linkage(dist_condensed, method="ward")

    return {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pc1_top_loadings": [{"word": w, "loading": round(float(l), 4)} for w, l in pc1_loadings],
        "pc2_top_loadings": [{"word": w, "loading": round(float(l), 4)} for w, l in pc2_loadings],
        "points": points,
        "ward_linkage": Z_link.tolist(),
    }


from stylometry.pos_stylometry import POSTagNGramAnalyzer
from stylometry.zeta import ZetaAnalyzer


class StylometricEnsembleAttributor:
    """
    Master suite pulling together all 12 stylometric methods into a unified
    training, cross-validation, and disputed-text attribution pipeline:
      1. Burrows's Classic Delta (MFW=150)
      2. Cosine / Würzburg Delta (MFW=150)
      3. Argamon's Quadratic Delta (MFW=150)
      4. Eder's Rank-Weighted Delta (MFW=150)
      5. Kilgarriff's Chi-Squared (MFW=500)
      6. Mendenhall's Characteristic Curves of Composition
      7. Mosteller & Wallace Function-Word Discriminant Classifier
      8. Character N-Gram Profile Classifier (3-grams & 4-grams)
      9. Morphosyntactic / POS Tag N-Gram Classifier (1/2/3-grams)
     10. Burrows / Craig Zeta (ζ) Distinctive Vocabulary Classifier
     11. Koppel & Winter's General Imposters Verification
     12. Composite Supervised ML Classifier (Function Words + Lexical + Punctuation/Syntax)
    """

    def __init__(self, n_mfw: int = 150, imposters_iterations: int = 100):
        self.n_mfw = n_mfw
        self.imposters_iterations = imposters_iterations

        self.delta_burrows = DeltaAnalyzer(n_mfw=n_mfw, metric="burrows", profile_mode="author_subcorpus")
        self.delta_cosine = DeltaAnalyzer(n_mfw=n_mfw, metric="cosine", profile_mode="document_centroid")
        self.delta_quadratic = DeltaAnalyzer(n_mfw=n_mfw, metric="quadratic", profile_mode="author_subcorpus")
        self.delta_eder = DeltaAnalyzer(n_mfw=n_mfw, metric="eder", profile_mode="author_subcorpus")
        self.kilgarriff = KilgarriffChiSquare(n_mfw=500)
        self.mendenhall = MendenhallCurveAnalyzer(max_word_length=16)
        self.mosteller_wallace = MostellerWallaceClassifier()
        self.char_ngram = CharNGramAnalyzer(n_values=(3, 4), top_k=500)
        self.pos_ngram = POSTagNGramAnalyzer(n_values=(1, 2, 3), top_k=250)
        self.zeta = ZetaAnalyzer(segment_size=1000, top_n_markers=50)
        self.imposters = GeneralImpostersVerifier(
            n_mfw=200, n_iterations=imposters_iterations, distance="cosine", random_state=42
        )
        self.composite_ml = CompositeStylometricClassifier(model_type="logreg", C=0.5)

    def fit(self, texts: Sequence[str], authors: Sequence[str]) -> "StylometricEnsembleAttributor":
        self.delta_burrows.fit(texts, authors)
        self.delta_cosine.fit(texts, authors)
        self.delta_quadratic.fit(texts, authors)
        self.delta_eder.fit(texts, authors)
        self.kilgarriff.fit(texts, authors)
        self.mendenhall.fit(texts, authors)
        self.mosteller_wallace.fit(texts, authors)
        self.char_ngram.fit(texts, authors)
        self.pos_ngram.fit(texts, authors)
        self.zeta.fit(texts, authors)
        self.imposters.fit(texts, authors)
        self.composite_ml.fit(texts, authors)
        return self

    def attribute_document(self, text: str) -> Dict[str, object]:
        """Run all 12 stylometric attribution/verification models on a single document."""
        b_pred, b_scores = self.delta_burrows.predict_one(text)
        c_pred, c_scores = self.delta_cosine.predict_one(text)
        q_pred, q_scores = self.delta_quadratic.predict_one(text)
        e_pred, e_scores = self.delta_eder.predict_one(text)
        k_pred, k_scores = self.kilgarriff.predict_one(text)
        m_pred, m_scores = self.mendenhall.predict_one(text)
        mw_pred, mw_probs = self.mosteller_wallace.predict_one(text)
        cng_pred, cng_scores = self.char_ngram.predict_one(text)
        pos_pred, pos_scores = self.pos_ngram.predict_one(text)
        zeta_pred, zeta_scores = self.zeta.predict_one(text)
        imp_pred, imp_scores = self.imposters.predict_one(text)
        ml_pred, ml_probs = self.composite_ml.predict_one(text)

        method_predictions = {
            "burrows_delta": str(b_pred),
            "cosine_delta": str(c_pred),
            "quadratic_delta": str(q_pred),
            "eder_delta": str(e_pred),
            "kilgarriff_chi2": str(k_pred),
            "mendenhall_curve": str(m_pred),
            "mosteller_wallace": str(mw_pred),
            "char_ngrams": str(cng_pred),
            "pos_ngrams": str(pos_pred),
            "craig_zeta": str(zeta_pred),
            "general_imposters": str(imp_pred),
            "composite_ml": str(ml_pred),
        }

        # Core 11 modern methods vote (excluding Mendenhall's 1887 word-length curve baseline)
        modern_methods = [
            "burrows_delta",
            "cosine_delta",
            "quadratic_delta",
            "eder_delta",
            "kilgarriff_chi2",
            "mosteller_wallace",
            "char_ngrams",
            "pos_ngrams",
            "craig_zeta",
            "general_imposters",
            "composite_ml",
        ]
        votes: Dict[str, int] = {}
        for m in modern_methods:
            ans = method_predictions[m]
            votes[ans] = votes.get(ans, 0) + 1

        consensus_author = max(votes, key=votes.get)
        agreement_ratio = votes[consensus_author] / float(len(modern_methods))

        return {
            "consensus_author": consensus_author,
            "consensus_agreement": round(agreement_ratio, 4),
            "vote_breakdown": votes,
            "predictions": method_predictions,
            "details": {
                "burrows_delta_distances": {k: round(v, 4) for k, v in b_scores.items()},
                "cosine_delta_distances": {k: round(v, 4) for k, v in c_scores.items()},
                "quadratic_delta_distances": {k: round(v, 4) for k, v in q_scores.items()},
                "eder_delta_distances": {k: round(v, 4) for k, v in e_scores.items()},
                "kilgarriff_chi2_distances": {k: round(v, 2) for k, v in k_scores.items()},
                "mendenhall_l1_distances": {k: round(v, 4) for k, v in m_scores.items()},
                "mosteller_wallace_probs": {k: round(v, 4) for k, v in mw_probs.items()},
                "char_ngram_cosine_distances": {k: round(v, 4) for k, v in cng_scores.items()},
                "pos_ngram_cosine_distances": {k: round(v, 4) for k, v in pos_scores.items()},
                "craig_zeta_net_scores": {k: round(v, 4) for k, v in zeta_scores.items()},
                "general_imposters_scores": {k: round(v, 4) for k, v in imp_scores.items()},
                "composite_ml_probs": {k: round(v, 4) for k, v in ml_probs.items()},
            },
        }


def evaluate_methods_cross_validation(
    texts: Sequence[str],
    authors: Sequence[str],
    n_splits: int = 5,
    n_mfw: int = 150,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all 12 stylometric methods using Stratified K-Fold (or Leave-One-Out if
    n_splits == -1) cross-validation on a corpus of undisputed texts.
    Returns accuracy and macro-F1 for each method and for the ensemble consensus.
    """
    texts_arr = list(texts)
    authors_arr = np.array(authors)

    if n_splits == -1 or n_splits >= len(texts_arr):
        splitter = LeaveOneOut().split(texts_arr)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
            texts_arr, authors_arr
        )

    method_names = [
        "burrows_delta",
        "cosine_delta",
        "quadratic_delta",
        "eder_delta",
        "kilgarriff_chi2",
        "mendenhall_curve",
        "mosteller_wallace",
        "char_ngrams",
        "pos_ngrams",
        "craig_zeta",
        "general_imposters",
        "composite_ml",
        "ensemble_consensus",
    ]
    y_true_all: List[str] = []
    y_pred_by_method: Dict[str, List[str]] = {m: [] for m in method_names}

    for train_idx, test_idx in splitter:
        train_texts = [texts_arr[i] for i in train_idx]
        train_authors = [str(authors_arr[i]) for i in train_idx]

        ensemble = StylometricEnsembleAttributor(n_mfw=n_mfw, imposters_iterations=60)
        ensemble.fit(train_texts, train_authors)

        for i in test_idx:
            y_true_all.append(str(authors_arr[i]))
            res = ensemble.attribute_document(texts_arr[i])
            for m, p in res["predictions"].items():
                y_pred_by_method[m].append(p)
            y_pred_by_method["ensemble_consensus"].append(str(res["consensus_author"]))

    metrics: Dict[str, Dict[str, float]] = {}
    for m in method_names:
        acc = accuracy_score(y_true_all, y_pred_by_method[m])
        f1 = f1_score(y_true_all, y_pred_by_method[m], average="macro")
        metrics[m] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "correct": int(sum(1 for t, p in zip(y_true_all, y_pred_by_method[m]) if t == p)),
            "total": len(y_true_all),
        }
    return metrics
