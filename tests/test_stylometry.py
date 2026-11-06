"""
Unit and integration tests for the `stylometry` package.
"""

from pathlib import Path
import pytest

from stylometry import (
    BootstrapConsensusAnalyzer,
    CharNGramAnalyzer,
    CompositeStylometricClassifier,
    DeltaAnalyzer,
    GeneralImpostersVerifier,
    KilgarriffChiSquare,
    MendenhallCurveAnalyzer,
    MostellerWallaceClassifier,
    POSTagNGramAnalyzer,
    RollingStylometryAnalyzer,
    StylometricEnsembleAttributor,
    UnmaskingVerifier,
    ZetaAnalyzer,
    compute_lexical_metrics,
    compute_syntactic_metrics,
    get_all_resources_summary,
    parse_federalist_corpus,
)

HAMILTON_SAMPLE_1 = (
    "Upon the whole, the power of raising revenue upon the national commerce is "
    "essential to the vigor of the federal government; while the state authorities "
    "retain enough jurisdiction over local policy to secure liberty. Upon every "
    "consideration of public safety, while we deliberate upon these national objects, "
    "there is enough evidence that vigor in the executive is requisite."
) * 12

HAMILTON_SAMPLE_2 = (
    "While we examine the commercial resources of the Union upon principles of "
    "public utility, we find enough reason to conclude that national credit "
    "depends upon a regular system of taxation; while any restriction upon that "
    "authority would destroy the vigor of the administration."
) * 12

MADISON_SAMPLE_1 = (
    "Whilst the powers delegated by the proposed Constitution are few and defined, "
    "those which remain in the State governments are numerous and indefinite; "
    "consequently no innovation on the Republican form can arise from the federal "
    "compact. Whilst each department has a constitutional will of its own, "
    "consequently the equilibrium on which liberty rests is preserved."
) * 12

MADISON_SAMPLE_2 = (
    "On a candid review of the federal system, whilst the several members retain "
    "their sovereignty on internal objects, consequently the danger of faction "
    "is diminished; whilst the authority on foreign relations is vested in the "
    "general council."
) * 12


def test_resources_catalog():
    summary = get_all_resources_summary()
    assert len(summary["methods"]) == 17
    assert len(summary["external_software"]) >= 5
    assert summary["word_lists"]["mosteller_wallace_markers_count"] == 30
    assert summary["word_lists"]["standard_function_words_count"] >= 140


def test_lexical_and_syntactic_metrics():
    lex = compute_lexical_metrics(HAMILTON_SAMPLE_1)
    assert lex["token_count"] > 100
    assert 0.0 < lex["ttr"] <= 1.0
    assert 0.0 < lex["mattr"] <= 1.0
    assert lex["yules_k"] > 0.0
    assert lex["root_ttr_guiraud"] > 0.0

    syn = compute_syntactic_metrics(HAMILTON_SAMPLE_1)
    assert syn["sentence_count"] >= 8
    assert syn["semicolon_rate_per_1k"] > 0.0
    assert syn["mean_sentence_length"] > 10.0


@pytest.mark.parametrize("metric", ["burrows", "cosine", "quadratic", "eder", "canberra", "minmax"])
def test_delta_analyzer_variants(metric):
    texts = [HAMILTON_SAMPLE_1, HAMILTON_SAMPLE_2, MADISON_SAMPLE_1, MADISON_SAMPLE_2]
    authors = ["Hamilton", "Hamilton", "Madison", "Madison"]

    analyzer = DeltaAnalyzer(n_mfw=40, metric=metric, profile_mode="author_subcorpus")
    analyzer.fit(texts, authors)

    pred_h, dists_h = analyzer.predict_one(HAMILTON_SAMPLE_1)
    pred_m, dists_m = analyzer.predict_one(MADISON_SAMPLE_1)

    assert pred_h == "Hamilton"
    assert pred_m == "Madison"
    assert dists_h["Hamilton"] < dists_h["Madison"]
    assert dists_m["Madison"] < dists_m["Hamilton"]


def test_explain_attribution():
    texts = [HAMILTON_SAMPLE_1, HAMILTON_SAMPLE_2, MADISON_SAMPLE_1, MADISON_SAMPLE_2]
    authors = ["Hamilton", "Hamilton", "Madison", "Madison"]
    analyzer = DeltaAnalyzer(n_mfw=40, metric="burrows")
    analyzer.fit(texts, authors)

    explanation = analyzer.explain_attribution(MADISON_SAMPLE_1, "Madison", "Hamilton", top_n=5)
    assert len(explanation) == 5
    words = {item["word"] for item in explanation}
    assert len(words) == 5


def test_kilgarriff_mendenhall_char_ngrams_imposters_and_ml():
    texts = [HAMILTON_SAMPLE_1, HAMILTON_SAMPLE_2, MADISON_SAMPLE_1, MADISON_SAMPLE_2] * 2
    authors = ["Hamilton", "Hamilton", "Madison", "Madison"] * 2

    kilg = KilgarriffChiSquare(n_mfw=50).fit(texts, authors)
    assert kilg.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"
    assert kilg.predict_one(MADISON_SAMPLE_1)[0] == "Madison"

    mend = MendenhallCurveAnalyzer().fit(texts, authors)
    assert mend.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"

    cng = CharNGramAnalyzer(top_k=100).fit(texts, authors)
    assert cng.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"
    assert cng.predict_one(MADISON_SAMPLE_1)[0] == "Madison"

    pos = POSTagNGramAnalyzer(top_k=60).fit(texts, authors)
    assert pos.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"
    assert pos.predict_one(MADISON_SAMPLE_1)[0] == "Madison"

    mw = MostellerWallaceClassifier().fit(texts, authors)
    assert mw.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"
    assert mw.predict_one(MADISON_SAMPLE_1)[0] == "Madison"

    imp = GeneralImpostersVerifier(n_mfw=30, n_iterations=30).fit(texts, authors)
    assert imp.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"
    assert imp.predict_one(MADISON_SAMPLE_1)[0] == "Madison"

    ml = CompositeStylometricClassifier(model_type="logreg").fit(texts, authors)
    assert ml.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"
    assert ml.predict_one(MADISON_SAMPLE_1)[0] == "Madison"


def test_zeta_rolling_bct_and_unmasking():
    texts = [HAMILTON_SAMPLE_1, HAMILTON_SAMPLE_2, MADISON_SAMPLE_1, MADISON_SAMPLE_2] * 2
    authors = ["Hamilton", "Hamilton", "Madison", "Madison"] * 2

    zeta = ZetaAnalyzer(segment_size=200, top_n_markers=15).fit(texts, authors)
    assert zeta.predict_one(HAMILTON_SAMPLE_1)[0] == "Hamilton"
    assert zeta.predict_one(MADISON_SAMPLE_1)[0] == "Madison"
    markers = zeta.get_contrastive_markers("Madison", top_k=5)
    assert len(markers["preferred_by_Madison"]) > 0

    roller = RollingStylometryAnalyzer(n_mfw=30, window_size=200, step_size=100).fit(texts, authors)
    mixed_text = HAMILTON_SAMPLE_1 + " " + MADISON_SAMPLE_1
    roll_res = roller.analyze_text(mixed_text)
    assert roll_res["num_windows"] >= 2
    assert roll_res["windows"][0]["winner"] == "Hamilton"
    assert roll_res["windows"][-1]["winner"] == "Madison"

    bct = BootstrapConsensusAnalyzer(mfw_grid=(20, 40), culling_grid=(0.0,)).fit(texts, authors)
    bct_res = bct.predict_consensus(MADISON_SAMPLE_1)
    assert bct_res["consensus_author"] == "Madison"
    assert bct_res["stability_score"] == 1.0

    unmask = UnmaskingVerifier(n_mfw=40, n_iterations=3, eliminate_per_side=2, chunk_size=120)
    deg = unmask.compute_degradation_curve([HAMILTON_SAMPLE_1, HAMILTON_SAMPLE_2], [MADISON_SAMPLE_1, MADISON_SAMPLE_2])
    assert len(deg["curve"]) >= 2


def test_federalist_corpus_parser():
    raw_path = Path("/home/antigravity/stylometry_toolkit/data/raw/pg1404.txt")
    if not raw_path.exists():
        pytest.skip("Federalist raw corpus not downloaded")
    docs = parse_federalist_corpus(raw_path)
    assert len(docs) == 85
    undisputed_h = [d for d in docs if d.author == "Hamilton"]
    undisputed_m = [d for d in docs if d.author == "Madison"]
    undisputed_j = [d for d in docs if d.author == "Jay"]
    disputed = [d for d in docs if d.status == "disputed"]
    joint = [d for d in docs if d.status == "joint"]

    assert len(undisputed_h) == 51
    assert len(undisputed_m) == 14
    assert len(undisputed_j) == 5
    assert len(disputed) == 12
    assert len(joint) == 3

    # Verify zero header leakage of author names in first 100 chars
    for d in docs:
        assert not d.text.startswith("HAMILTON")
        assert not d.text.startswith("MADISON")
        assert not d.text.startswith("JAY")
        assert "To the People of the State of New York" not in d.text[:120]
