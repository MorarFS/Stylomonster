/* ==========================================================================
   Computational Stylometry & Authorship Attribution — Dataset & Pedagogical Store
   ========================================================================== */

window.STYLOMETRY_DATA = {
  methods: [
    {
      id: "burrows_delta",
      name: "Burrows's Classic Delta (Δ_B)",
      category: "Delta Family",
      year: 2002,
      authors: "John F. Burrows",
      module: "stylometry.delta.DeltaAnalyzer(metric='burrows')",
      formula: "Δ_B(D₁, D₂) = (1 / M) · Σ_{i=1..M} | z_i(D₁) − z_i(D₂) |,   where z_i(D) = (f_i(D) − μ_i) / σ_i",
      intuition: "The foundational distance measure of modern computational stylometry. In raw frequency space, the top 5 words ('the', 'of', 'to', 'and', 'in') have huge magnitudes and drown out subtle function words at rank 80 ('upon', 'whilst', 'enough'). Burrows solved this by converting each word's relative frequency into a corpus z-score (mean 0, standard deviation 1) before taking the mean absolute (Manhattan) distance.",
      useCases: [
        "Closed-set authorship attribution among 2 to 20 candidate authors when training texts are ≥2,500 words each.",
        "Works best with M = 100 to 300 Most Frequent Words (MFWs), which are dominated by topic-independent grammatical function words.",
        "Highly interpretable: each word's absolute z-difference |z_i(D₁) − z_i(D₂)| directly quantifies its additive contribution to stylistic distance."
      ],
      limits: [
        "Sensitive to extreme text-length disparities: short texts (<1,000 words) have many zero counts that produce noisy negative z-scores.",
        "Vulnerable to outlier z-scores when M > 500 (rare content words with tiny corpus standard deviations σ_i can blow up |z_i|).",
        "Closed-set fallacy: always assigns the minimum-distance candidate even if the true author is not in the reference pool."
      ],
      code: `from stylometry import DeltaAnalyzer\n\nanalyzer = DeltaAnalyzer(n_mfw=150, metric="burrows", profile_mode="author_subcorpus")\nanalyzer.fit(train_texts, train_authors)\npred_author, distances = analyzer.predict_one(disputed_text)\ntop_drivers = analyzer.explain_attribution(disputed_text, "Madison", "Hamilton", top_n=10)`
    },
    {
      id: "cosine_wurzburg_delta",
      name: "Cosine / Würzburg Delta (Δ_cos)",
      category: "Delta Family",
      year: 2017,
      authors: "Peter W. H. Smith & W. Aldridge (2011); Stefan Evert et al. (2017)",
      module: "stylometry.delta.DeltaAnalyzer(metric='cosine')",
      formula: "Δ_cos(D₁, D₂) = 1 − [ z(D₁) · z(D₂) ] / ( ‖z(D₁)‖₂ · ‖z(D₂)‖₂ )",
      intuition: "Projects each document's z-score vector onto the unit hypersphere and measures the cosine angle between them. Demonstrated by Evert et al. (2017) across English, German, and French corpora to be the single most robust variant of Delta.",
      useCases: [
        "Recommended default metric for modern stylometric clustering, PCA/dendrograms, and medium-to-long literary attribution.",
        "Remains stable even at large feature sizes (M = 500 to 3,000 MFWs) because L2 vector normalization prevents single outlier features from dominating.",
        "Focuses on the directional pattern of over-use and under-use across the vocabulary rather than raw deviation amplitude."
      ],
      limits: [
        "When a candidate author has very few training documents or highly imbalanced subcorpus sizes, document-centroid vs. pooled-subcorpus standardization must be chosen carefully.",
        "Still a closed-set distance metric: requires pairing with General Imposters (GI) for open-set verification."
      ],
      code: `from stylometry import DeltaAnalyzer\n\ncosine_delta = DeltaAnalyzer(n_mfw=250, metric="cosine", profile_mode="document_centroid")\ncosine_delta.fit(train_texts, train_authors)\npred_author, cos_distances = cosine_delta.predict_one(disputed_text)`
    },
    {
      id: "argamon_quadratic_delta",
      name: "Argamon's Quadratic Delta (Δ_Q)",
      category: "Delta Family",
      year: 2008,
      authors: "Shlomo Argamon",
      module: "stylometry.delta.DeltaAnalyzer(metric='quadratic')",
      formula: "Δ_Q(D₁, D₂) = √[ (1 / M) · Σ_{i=1..M} ( z_i(D₁) − z_i(D₂) )² ]",
      intuition: "Provides a formal probabilistic interpretation of Burrows's Delta: if word frequencies are modeled as independent Gaussian random variables around each author's mean, maximum log-likelihood attribution corresponds to minimizing the Euclidean (L2) distance on z-scores rather than the Manhattan (L1) distance.",
      useCases: [
        "Penalizes large single-feature stylistic discrepancies much more heavily than Burrows's L1 Delta.",
        "Achieved 93.85% (61/65) cross-validation accuracy on the undisputed Federalist Papers and 100% (45/45) on our 19th-Century Literature corpus."
      ],
      limits: [
        "Word frequencies in natural language follow heavy-tailed (Zipfian / Negative Binomial) distributions rather than pure Gaussians, making L2 sensitive to topical spikes at large M."
      ],
      code: `from stylometry import DeltaAnalyzer\n\nquad_delta = DeltaAnalyzer(n_mfw=150, metric="quadratic")\nquad_delta.fit(train_texts, train_authors)\npred_author, dists = quad_delta.predict_one(disputed_text)`
    },
    {
      id: "eder_delta",
      name: "Eder's Rank-Weighted Delta (Δ_E)",
      category: "Delta Family",
      year: 2016,
      authors: "Maciej Eder, Jan Rybicki & Mike Kestemont",
      module: "stylometry.delta.DeltaAnalyzer(metric='eder')",
      formula: "Δ_E(D₁, D₂) = Σ_{i=1..M} w_i · | z_i(D₁) − z_i(D₂) |,   where w_i = (M − i + 1) / Σ_k k",
      intuition: "In standard Burrows's Delta, the 1st most frequent word ('the') and the 150th most frequent word have identical weight after z-score standardization. Eder's Delta multiplies each feature by an inverse rank weight so that high-frequency, topic-invariant function words exert stronger influence than lower-ranked words.",
      useCases: [
        "Particularly effective for shorter texts (~1,500–5,000 words) and highly inflected languages where lower-ranked MFWs begin to include topical nouns.",
        "Tied for top Delta variant on the Federalist Papers (93.85% 2-class CV, 92.86% 3-class CV) and 100% on 19th-Century Literature."
      ],
      limits: [
        "Linear rank decay is heuristic; if an author's most distinctive habit lies in mid-frequency markers (ranks 100–300), Eder's weight down-scales them."
      ],
      code: `from stylometry import DeltaAnalyzer\n\neder_delta = DeltaAnalyzer(n_mfw=150, metric="eder")\neder_delta.fit(train_texts, train_authors)\npred_author, dists = eder_delta.predict_one(disputed_text)`
    },
    {
      id: "canberra_minmax_delta",
      name: "Canberra & MinMax (Ruzicka) Distances",
      category: "Delta Family",
      year: 2014,
      authors: "Moshe Koppel & Yaron Winter (2014); Mike Kestemont et al. (2016)",
      module: "stylometry.delta.DeltaAnalyzer(metric='minmax')",
      formula: "Δ_MinMax(A, B) = 1 − [ Σ_{i=1..M} min(x_{i,A}, x_{i,B}) ] / [ Σ_{i=1..M} max(x_{i,A}, x_{i,B}) ]",
      intuition: "Non-parametric ratio distances borrowed from ecology (Ruzicka / Bray-Curtis / Canberra). Instead of relying on Gaussian mean and variance estimates, MinMax compares the intersection over union of non-negative feature profiles.",
      useCases: [
        "Standard base metric inside Koppel's General Imposters (GI) verification system and sparse character n-gram vectors.",
        "Bounded strictly in [0, 1] and naturally handles zero-inflated feature vectors without division-by-zero variance blowups."
      ],
      limits: [
        "Without z-score scaling or TF-IDF weighting, raw MinMax can be dominated by the top 20 most frequent words unless features are pre-standardized."
      ],
      code: `from stylometry import DeltaAnalyzer\n\nminmax = DeltaAnalyzer(n_mfw=200, metric="minmax")\nminmax.fit(train_texts, train_authors)\npred, scores = minmax.predict_one(disputed_text)`
    },
    {
      id: "mosteller_wallace_discriminant",
      name: "Mosteller & Wallace Function-Word Discriminant",
      category: "Function Words & Zeta",
      year: 1964,
      authors: "Frederick Mosteller & David L. Wallace",
      module: "stylometry.classifiers.MostellerWallaceClassifier",
      formula: "Log-Odds / L2-Penalized Discriminant over 30 non-contextual marker words ('upon', 'whilst', 'while', 'enough', 'consequently', ...)",
      intuition: "In their landmark 1964 book 'Inference and Disputed Authorship: The Federalist', Mosteller and Wallace screened out all political content words ('court', 'war', 'tax', 'senate') and focused on 30 function words whose rates depend purely on unconscious habit: e.g., Hamilton writes 'upon' 3.27 times per 1,000 words while Madison writes it only 0.16 times per 1,000 words; Hamilton writes 'while' whereas Madison writes 'whilst'.",
      useCases: [
        "Gold-standard discriminant when two or more authors write in the exact same genre, time period, and subject matter.",
        "Achieved 100.0% (65/65) cross-validation accuracy separating Hamilton and Madison on the undisputed Federalist Papers."
      ],
      limits: [
        "Curated marker lists are partly period- and language-specific (e.g., 18th-century 'whilst' vs. 'while' is less active in 21st-century prose).",
        "Because function words are easy to inspect, an adversary who knows the specific marker list could consciously search-and-replace 'upon' with 'on'."
      ],
      code: `from stylometry import MostellerWallaceClassifier\n\nmw = MostellerWallaceClassifier().fit(train_texts, train_authors)\npred_author, posterior_probs = mw.predict_one(disputed_text)`
    },
    {
      id: "craig_burrows_zeta",
      name: "Burrows's & Craig's Zeta (ζ) Distinctive Vocabulary",
      category: "Function Words & Zeta",
      year: 2007,
      authors: "John F. Burrows (2007); Hugh Craig & Arthur F. Kinney (2009)",
      module: "stylometry.zeta.ZetaAnalyzer",
      formula: "ζ(w) = d_A(w) − d_B(w) ∈ [−1, +1],   where d_A(w) = fraction of Author A's segments containing word w",
      intuition: "While Delta focuses on the top 150 high-frequency grammar words, Zeta (ζ) explores the middle strata of the vocabulary. By splitting each author's corpus into equal-sized segments (e.g. 1,000 words) and measuring document dispersion rather than raw count, Zeta finds words consistently favored by Author A and consistently avoided by Author B without being skewed by a single topical chapter.",
      useCases: [
        "Discovering interpretable lexical contrasts between two authors (e.g., Shakespeare vs. Marlowe, or Hamilton's 'upon/kind/vigor' vs. Madison's 'themselves/among/fully').",
        "Visualizing two-dimensional Zeta separation planes (positive marker density vs. negative marker density)."
      ],
      limits: [
        "Because Zeta reaches into mid-frequency vocabulary, genre or topic differences between Author A and Author B can leak into top Zeta words if corpora are not genre-matched.",
        "Requires enough text per author to form at least 10–20 segments."
      ],
      code: `from stylometry import ZetaAnalyzer\n\nzeta = ZetaAnalyzer(segment_size=1000, top_n_markers=50).fit(train_texts, train_authors)\ncontrast = zeta.get_contrastive_markers("Madison", top_k=15)\npred, net_scores = zeta.predict_one(disputed_text)`
    },
    {
      id: "kilgarriff_chi_squared",
      name: "Kilgarriff's Chi-Squared (χ²) Distance",
      category: "Delta Family",
      year: 2001,
      authors: "Adam Kilgarriff",
      module: "stylometry.chi_square.KilgarriffChiSquare",
      formula: "χ²(A, B) = Σ_{w ∈ MFW} [ (O_{w,A} − E_{w,A})² / E_{w,A} + (O_{w,B} − E_{w,B})² / E_{w,B} ]",
      intuition: "Treats authorship attribution as a goodness-of-fit homogeneity test between two corpora. Pools the candidate author's known corpus A and the unknown text B to compute expected word counts under the null hypothesis that both came from the same linguistic generator, then sums Pearson's χ² across the top 500 words.",
      useCases: [
        "Comparing a test document against large, pooled authorial subcorpora without needing multiple separate documents per author to estimate standard deviations.",
        "Naturally weights words by their statistical surprise relative to expected count."
      ],
      limits: [
        "Word tokens in natural text exhibit 'burstiness' (clumping within documents), violating the binomial independence assumption of Pearson's χ²—so raw χ² p-values cannot be used directly as authorship probabilities; only relative χ² rankings across candidates are valid.",
        "Sensitive to large subcorpus size imbalances (e.g., Hamilton's 112k words vs. Jay's 8.3k words)."
      ],
      code: `from stylometry import KilgarriffChiSquare\n\nchi2 = KilgarriffChiSquare(n_mfw=500).fit(train_texts, train_authors)\npred_author, chi2_scores = chi2.predict_one(disputed_text)`
    },
    {
      id: "mendenhall_curves",
      name: "Mendenhall's Characteristic Curves of Composition",
      category: "Lexical Richness & Curves",
      year: 1887,
      authors: "T. C. Mendenhall",
      module: "stylometry.lexical.MendenhallCurveAnalyzer",
      formula: "P(|w| = k) = (1 / N) · Σ_{t=1..N} I( length(w_t) = k )   for k = 1, 2, ..., 16",
      intuition: "Proposed by physicist T. C. Mendenhall in 1887 (and applied in 1901 to Shakespeare vs. Bacon using human counters!), this is the earliest quantitative stylometric method. It plots the relative frequency of 1-letter, 2-letter, 3-letter, ..., 16-letter words.",
      useCases: [
        "Historical baseline and pedagogical visualization of Anglo-Saxon (short 3–4 letter words) vs. Latinate/Romance (long 7–12 letter words) prose register.",
        "Distinguishing broad genre shifts or authors with starkly different educational/lexical registers."
      ],
      limits: [
        "Low discriminative power when two authors write in the same genre and era (e.g., Hamilton's mean word length is 4.80 letters vs. Madison's 4.90 letters—their Mendenhall curves nearly overlap!).",
        "Throws away all lexical identity and syntax: 'upon' and 'with' are both 4-letter words."
      ],
      code: `from stylometry import MendenhallCurveAnalyzer\n\nmend = MendenhallCurveAnalyzer(max_word_length=16).fit(train_texts, train_authors)\ncurve = mend.compute_curve(disputed_text)\npred, l1_dists = mend.predict_one(disputed_text)`
    },
    {
      id: "lexical_diversity_suite",
      name: "Lexical Richness & Vocabulary Diversity Suite (MATTR, Yule's K, Honoré's R)",
      category: "Lexical Richness & Curves",
      year: 1944,
      authors: "G. U. Yule (1944), P. Guiraud (1954), A. Honoré (1979), Covington & McFall (2010)",
      module: "stylometry.lexical.compute_lexical_metrics",
      formula: "MATTR(W) = mean_{t} (V_{t..t+W} / W);   Yule's K = 10⁴ · (Σ i² V_i − N) / N²;   Honoré's R = 100 · ln(N) / (1 − V₁/V)",
      intuition: "Measures how rich, varied, or repetitive an author's vocabulary is. Raw Type-Token Ratio (TTR = V/N) is fatally flawed because it automatically shrinks as a text gets longer (Heaps' Law). Modern stylometry uses Moving-Average TTR (MATTR over a sliding 500-word window), Yule's K Characteristic (word repetition rate), and Honoré's R (scaling of hapax legomena V₁, words used only once).",
      useCases: [
        "Quantifying vocabulary breadth and repetition habits completely independent of document length (via MATTR and Yule's K).",
        "Strong supporting feature set inside multivariate classifiers (e.g., Charlotte Brontë's MATTR = 0.543 & Honoré's R = 2550 vs. Jane Austen's MATTR = 0.495 & Honoré's R = 2020)."
      ],
      limits: [
        "Scalar summary statistics alone cannot identify an author among many candidates without word-level features; two different authors can share a similar vocabulary size.",
        "Raw TTR, Guiraud's R, and Brunét's W should never be compared directly between a 500-word letter and a 100,000-word novel—always use MATTR or equal-sized chunks."
      ],
      code: `from stylometry import compute_lexical_metrics\n\nmetrics = compute_lexical_metrics(text, mattr_window=500)\nprint(metrics["mattr"], metrics["yules_k"], metrics["honores_r"])`
    },
    {
      id: "syntactic_punctuation_fingerprint",
      name: "Syntactic Rhythm, Punctuation Fingerprints & Readability",
      category: "Syntactic & POS Sequences",
      year: 1998,
      authors: "E. Stamatatos et al. (2000); H. Baayen et al. (1996)",
      module: "stylometry.syntactic.compute_syntactic_metrics",
      formula: "Punctuation rates per 1,000 words (; : , — ? ! ( ) \"\") + Sentence Length (μ, σ, Median, IQR) + 5 Readability Indices",
      intuition: "How an author paces clauses and punctuates pauses—especially their habitual use of semicolons (;), colons (:), em-dashes (—), and parentheticals—is deeply ingrained and largely invisible to conscious imitation. For instance, in our 19th-Century corpus, Charlotte Brontë uses 18.04 semicolons per 1,000 words while Sir Arthur Conan Doyle uses only 1.22!",
      useCases: [
        "Distinguishing authors with distinct clause-linking architectures (parataxis vs. hypotaxis) and sentence-length rhythms.",
        "Crucial signal when lexical vocabulary is constrained by technical or legal subject matter."
      ],
      limits: [
        "Historically sensitive to copy-editing, typesetting house style, and modernizations (e.g., 18th-century printers sometimes inserted or standardized commas and semicolons).",
        "Dialogue-heavy fiction inflates quotation marks, question marks, and short sentences compared to third-person exposition by the same author."
      ],
      code: `from stylometry import compute_syntactic_metrics\n\nsyn = compute_syntactic_metrics(text)\nprint(syn["semicolon_rate_per_1k"], syn["mean_sentence_length"], syn["flesch_kincaid_grade"])`
    },
    {
      id: "char_ngram_profiles",
      name: "Character N-Gram Profile Attribution (CNG)",
      category: "Syntactic & POS Sequences",
      year: 2003,
      authors: "Vlado Kešelj et al. (2003); Efstathios Stamatatos (2009)",
      module: "stylometry.syntactic.CharNGramAnalyzer",
      formula: "Cosine Delta over standardized frequencies of the top-K character 3-grams and 4-grams",
      intuition: "Slides a 3- or 4-character window across the raw text (including spaces and punctuation). Without needing any language-specific tokenizer or stemmer, character n-grams simultaneously capture short function words (' th', ' of '), derivational suffixes ('tion', 'ment', 'ing '), verb tense markers ('ed '), and punctuation-space boundaries ('; th').",
      useCases: [
        "Language-independent stylometry and noisy text (OCR errors, social media, historical spelling variations, agglutinative languages).",
        "Achieved 100.0% (45/45) cross-validation accuracy across our 5-author 19th-Century Literature benchmark."
      ],
      limits: [
        "Character n-grams of length ≥4 can capture stems of content/thematic words (e.g., 'cons' from 'constitution', 'repr' from 'representative'), making them slightly more topic-sensitive than pure function words on small corpora.",
        "Less directly interpretable to human readers than whole-word markers."
      ],
      code: `from stylometry import CharNGramAnalyzer\n\ncng = CharNGramAnalyzer(n_values=(3, 4), top_k=500).fit(train_texts, train_authors)\npred_author, cos_dists = cng.predict_one(disputed_text)`
    },
    {
      id: "pos_ngram_stylometry",
      name: "Morphosyntactic & POS Tag N-Gram Transitions",
      category: "Syntactic & POS Sequences",
      year: 2007,
      authors: "Graeme Hirst & Ol'ga Feiguina (2007); Efstathios Stamatatos (2009)",
      module: "stylometry.pos_stylometry.POSTagNGramAnalyzer",
      formula: "Cosine Delta over standardized 1-gram, 2-gram, and 3-gram frequencies of morphosyntactic category sequences",
      intuition: "Replaces every word with its grammatical/morphological category (e.g. PREP → DET → ADJ_DERIV → NOUN_DERIV → P_CLAUSE) and measures transition frequencies. This strips away surface vocabulary and captures the syntactic 'skeleton' of how an author constructs phrases.",
      useCases: [
        "Cross-topic attribution where an author writes about completely different subjects in training vs. test documents.",
        "Detecting syntactic calques, translationese, or structural habits."
      ],
      limits: [
        "Coarser feature space (16–36 tags) than lexical MFWs, so POS n-grams usually have ~5–10% lower standalone accuracy than function-word Delta and work best inside an ensemble."
      ],
      code: `from stylometry import POSTagNGramAnalyzer\n\npos = POSTagNGramAnalyzer(n_values=(1, 2, 3), top_k=250).fit(train_texts, train_authors)\npred_author, dists = pos.predict_one(disputed_text)`
    },
    {
      id: "koppel_general_imposters",
      name: "Koppel & Winter's General Imposters (GI) Verification",
      category: "Open-Set Verification",
      year: 2014,
      authors: "Moshe Koppel & Yaron Winter (2014); Mike Kestemont et al. (2016)",
      module: "stylometry.imposters.GeneralImpostersVerifier",
      formula: "S(X, C) = (1 / K) · Σ_{k=1..K} I[ min_{c ∈ C} d_{F_k}(X, c) < min_{imp ∈ I_k} d_{F_k}(X, imp) ] ∈ [0, 1]",
      intuition: "Solves the 'Police Lineup' (Open-Set) problem: What if NONE of your candidate authors wrote the disputed text? Across K = 100 Monte Carlo trials, GI randomly drops 50% of the features and pits the suspected author C against a randomly sampled lineup of distractor ('imposter') authors. Only if X is consistently closer to C than to all imposters across random feature subspaces (S > 0.50–0.70) is authorship verified.",
      useCases: [
        "Mandatory whenever the true author might not be among the primary suspects (open-set verification, forensic linguistics, disputed historical attributions).",
        "Verified all 12 disputed Federalist Papers for Madison (scores 0.60 to 0.94) and Federalist No. 64 for John Jay (score 0.93)."
      ],
      limits: [
        "Requires a well-matched pool of imposter texts from the same language, genre, and historical era; if imposters are from a different era, the candidate will win trivially.",
        "Scores in the 'grey zone' (0.35 < S < 0.65, such as Federalist No. 55 at S = 0.64) indicate potential co-authorship, short sample length, or ambiguous attribution."
      ],
      code: `from stylometry import GeneralImpostersVerifier\n\ngi = GeneralImpostersVerifier(n_mfw=200, n_iterations=100, distance="cosine")\ngi.fit(reference_texts, reference_authors)\nwinner, verification_scores = gi.predict_one(disputed_text)`
    },
    {
      id: "koppel_unmasking",
      name: "Koppel & Schler's 'Unmasking' Degradation Curves",
      category: "Open-Set Verification",
      year: 2004,
      authors: "Moshe Koppel & Jonathan Schler",
      module: "stylometry.zeta.UnmaskingVerifier",
      formula: "Iterative k-fold CV accuracy curve A_0, A_1, ..., A_R after removing top-2m separating features per round",
      intuition: "Measures the 'depth of difference' between two long texts A and B. Even two books by the SAME author can be separated with ~90% accuracy by a linear classifier using a dozen character names or setting words. Unmasking iteratively trains a classifier, deletes the most strongly weighted positive and negative words, and repeats. If A and B are by the SAME author, accuracy plummets within 3–5 rounds; if by DIFFERENT authors, accuracy stays high across many rounds.",
      useCases: [
        "Verifying whether two substantial corpora or books were written by the same person despite differing topics or genres.",
        "In our Federalist benchmark, Disputed-vs-Hamilton stayed at 86.9% mean accuracy across 8 rounds (different authors), whereas Disputed-vs-Madison collapsed to 41.7% (same author!)."
      ],
      limits: [
        "Requires relatively long texts (at least 5,000–10,000 words per side) so each side can be partitioned into ≥10 chunks for internal cross-validation."
      ],
      code: `from stylometry import UnmaskingVerifier\n\nunmask = UnmaskingVerifier(n_mfw=200, n_iterations=8, eliminate_per_side=4)\ncurve_info = unmask.compute_degradation_curve(disputed_texts, madison_texts)`
    },
    {
      id: "eder_rolling_stylometry",
      name: "Eder's Rolling Stylometry (Sequential Collaborative Analysis)",
      category: "Sequential & Consensus",
      year: 2016,
      authors: "Maciej Eder",
      module: "stylometry.rolling.RollingStylometryAnalyzer",
      formula: "d_t(c) = Δ( Window(X, start=t·S, len=L), Author_c )   across overlapping windows t = 0, 1, ..., T",
      intuition: "Traditional stylometry treats a document as a single homogeneous bag of words—which breaks down when a text was co-authored, edited by a second hand, or interpolated over time. Rolling Stylometry slides an overlapping window (e.g. 500–2,000 words) from the beginning to the end of a work and plots which author dominates each passage.",
      useCases: [
        "Detecting collaborative authorship transitions, ghostwritten chapters, editorial interpolations, or mixed-hand essays (such as Federalist Nos. 18–20).",
        "Verifying whether a disputed document is stylistically uniform throughout or a composite draft."
      ],
      limits: [
        "Because individual windows are necessarily short (500–1,500 words), per-window variance is higher than whole-document attribution; short spikes should not be over-interpreted."
      ],
      code: `from stylometry import RollingStylometryAnalyzer\n\nroller = RollingStylometryAnalyzer(n_mfw=100, window_size=500, step_size=150)\nroller.fit(train_texts, train_authors)\ntrajectory = roller.analyze_text(collaborative_text)`
    },
    {
      id: "eder_bootstrap_consensus",
      name: "Bootstrap Consensus Trees (BCT) & Multi-Metric Ensemble",
      category: "Sequential & Consensus",
      year: 2012,
      authors: "Maciej Eder (2012, 2016); Jan Rybicki; Efstathios Stamatatos (2009)",
      module: "stylometry.consensus_tree.BootstrapConsensusAnalyzer / StylometricEnsembleAttributor",
      formula: "Stability(X, C) = (1 / |G|) · Σ_{(M, cull, metric) ∈ G} I[ argmin_a Δ_{M,cull,metric}(X, a) = C ]",
      intuition: "A single stylometric test at an arbitrary cutoff (say, M = 100 MFWs with Burrows's Delta) can be brittle or subject to researcher cherry-picking. Bootstrap Consensus evaluates the text across 48 parameter combinations (MFW ∈ {50..500} × Culling ∈ {0%, 20%} × 3 Delta metrics) plus our 12-algorithm ensemble (combining Delta, χ², Mosteller-Wallace, Character/POS n-grams, Zeta, Imposters, and Supervised ML).",
      useCases: [
        "Preventing hyperparameter p-hacking and quantifying attribution stability.",
        "11 of the 12 disputed Federalist Papers achieve 48/48 (100.0%) stability across the entire parameter grid, while Federalist No. 55 shows 30/48 (62.5%), transparently flagging its mixed signal."
      ],
      limits: [
        "Consensus across correlated word-frequency methods still relies on the quality and genre-matching of the underlying reference corpus; if the reference corpus is biased, all methods in the ensemble can share that bias."
      ],
      code: `from stylometry import StylometricEnsembleAttributor, BootstrapConsensusAnalyzer\n\nens = StylometricEnsembleAttributor(n_mfw=150).fit(train_texts, train_authors)\nreport = ens.attribute_document(disputed_text)\n\nbct = BootstrapConsensusAnalyzer().fit(train_texts, train_authors)\nstability = bct.predict_consensus(disputed_text)`
    }
  ],

  presets: {
    fed_51: {
      label: "Federalist No. 51 (Disputed → Madison)",
      context: "Federalist",
      trueStatus: "Disputed (Historically & Stylometrically attributed to James Madison)",
      text: `To what expedient, then, shall we finally resort, for maintaining in practice the necessary partition of power among the several departments, as laid down in the Constitution? The only answer that can be given is, that as all these exterior provisions are found to be inadequate, the defect must be supplied, by so contriving the interior structure of the government as that its several constituent parts may, by their mutual relations, be the means of keeping each other in their proper places. Without presuming to undertake a full development of this important idea, I will hazard a few general observations, which may perhaps place it in a clearer light, and enable us to form a more correct judgment of the principles and structure of the government planned by the convention. In order to lay a due foundation for that separate and distinct exercise of the different powers of government, which to a certain extent is admitted on all hands to be essential to the preservation of liberty, it is evident that each department should have a will of its own; and consequently should be so constituted that the members of each should have as little agency as possible in the appointment of the members of the others. Were this principle rigorously adhered to, it would require that all the appointments for the supreme executive, legislative, and judiciary magistracies should be drawn from the same fountain of authority, the people, through channels having no communication whatever with one another. Perhaps such a plan of constructing the several departments would be less difficult in practice than it may in contemplation appear. Some difficulties, however, and some additional expense would attend the execution of it. Some deviations, therefore, from the principle must be admitted. In the constitution of the judiciary department in particular, it might be inexpedient to insist rigorously on the principle: first, because peculiar qualifications being essential in the members, the primary consideration ought to be to select that mode of choice which best secures these qualifications; secondly, because the permanent tenure by which the appointments are held in that department, must soon destroy all sense of dependence on the authority conferring them. It is equally evident, that the members of each department should be as little dependent as possible on those of the others, for the emoluments annexed to their offices. Were the executive magistrate, or the judges, not independent of the legislature in this particular, their independence in every other would be merely nominal. But the great security against a gradual concentration of the several powers in the same department, consists in giving to those who administer each department the necessary constitutional means and personal motives to resist encroachments of the others. The provision for defense must in this, as in all other cases, be made commensurate to the danger of attack. Ambition must be made to counteract ambition. The interest of the man must be connected with the constitutional rights of the place. It may be a reflection on human nature, that such devices should be necessary to control the abuses of government. But what is government itself, but the greatest of all reflections on human nature? If men were angels, no government would be necessary. If angels were to govern men, neither external nor internal controls on government would be necessary. In framing a government which is to be administered by men over men, the great difficulty lies in this: you must first enable the government to control the governed; and in the next place oblige it to control itself. A dependence on the people is, no doubt, the primary control on the government; but experience has taught mankind the necessity of auxiliary precautions. Whilst all authority in it will be derived from and dependent on the society, the society itself will be broken into so many parts, interests, and classes of citizens, that the rights of individuals, or of the minority, will be in little danger from interested combinations of the majority.`
    },
    fed_01: {
      label: "Federalist No. 1 (Alexander Hamilton)",
      context: "Federalist",
      trueStatus: "Undisputed Alexander Hamilton (Note frequent 'upon', 'while', 'enough', 'vigor')",
      text: `After an unequivocal experience of the inefficacy of the subsisting federal government, you are called upon to deliberate on a new Constitution for the United States of America. The subject speaks its own importance; comprehending in its consequences nothing less than the existence of the UNION, the safety and welfare of the parts of which it is composed, the fate of an empire in many respects the most interesting in the world. It has been frequently remarked that it seems to have been reserved to the people of this country, by their conduct and example, to decide the important question, whether societies of men are really capable or not of establishing good government from reflection and choice, or whether they are forever destined to depend for their political constitutions on accident and force. If there be any truth in the remark, the crisis at which we are arrived may with propriety be regarded as the era in which that decision is to be made; and a wrong election of the part we shall act may, in this view, deserve to be considered as the general misfortune of mankind. This idea will add the inducements of philanthropy to those of patriotism, to heighten the solicitude which all considerate and good men must feel for the event. Happy will it be if our choice should be directed by a judicious estimate of our true interests, unperplexed and unbiased by considerations not connected with the public good. But this is a thing more ardently to be wished than seriously to be expected. The plan offered to our deliberations affects too many particular interests, innovates upon too many local institutions, not to involve in its discussion a variety of objects foreign to its merits, and of views, passions and prejudices little favorable to the discovery of truth. Among the most formidable of the obstacles which the new Constitution will have to encounter may readily be distinguished the obvious interest of a certain class of men in every State to resist all changes which may hazard a diminution of the power, emolument, and consequence of the offices they hold under the State establishments; and the perverted ambition of another class of men, who will either hope to aggrandize themselves by the confusions of their country, or will flatter themselves with fairer prospects of elevation from the subdivision of the empire into several partial confederacies than from its union under one government. It is not, however, my design to dwell upon observations of this kind. I am well aware that it would be disingenuous to resolve indiscriminately the opposition of any set of men (merely because their situations might subject them to suspicion) into interested or ambitious views. Candor will oblige us to admit that even such men may be actuated by upright intentions; and it cannot be doubted that much of the opposition which has made its appearance, or may hereafter make its appearance, will spring from sources, blameless at least, if not respectable—the honest errors of minds led astray by preconceived jealousies and fears. So numerous indeed and so powerful are the causes which serve to give a false bias to the judgment, that we, upon many occasions, see wise and good men on the wrong as well as on the right side of questions of the first magnitude to society. And a further reason for caution, in this respect, might be drawn from the reflection that we are not always sure that those who advocate the truth are influenced by purer principles than their antagonists. Ambition, avarice, personal animosity, party opposition, and many other motives not more laudable than these, are apt to operate as well upon those who support as those who oppose the right side of a question. Were there not even these inducements to moderation, nothing could be more ill-judged than that intolerant spirit which has, at all times, characterized political parties. For in politics, as in religion, it is equally absurd to aim at making proselytes by fire and sword. Heresies in either can rarely be cured by persecution. And yet, however just these sentiments will be allowed to be, we have already sufficient indications that it will happen in this as in all former cases of great national discussion. A torrent of angry and malignant passions will be let loose. To judge from the conduct of the opposite parties, we shall be led to conclude that they will mutually hope to evince the justness of their opinions, and to increase the number of their converts by the loudness of their declamations and the bitterness of their invectives. An enlightened zeal for the energy and efficiency of government will be stigmatized as the offspring of a temper fond of despotic power and hostile to the principles of liberty. It will beforgotten, on the one hand, that jealousy is the usual concomitant of love, and that the noble enthusiasm of liberty is apt to be infected with a spirit of narrow and illiberal distrust. On the other hand, it will be equally forgotten that the vigor of government is essential to the security of liberty; that, in the contemplation of a sound and well-informed judgment, their interest can never be separated; and that a dangerous ambition more often lurks behind the specious mask of zeal for the rights of the people than under the forbidding appearance of zeal for the firmness and efficiency of government. History will teach us that the former has been found a much more certain road to the introduction of despotism than the latter, and that of those men who have overturned the liberties of republics, the greatest number have begun their career by paying an obsequious court to the people; commencing demagogues, and ending tyrants. In the course of the preceding observations, I have had an eye, my fellow-citizens, to putting you upon your guard against all attempts, from whatever quarter, to influence your decision in a matter of the utmost moment to your welfare, by any impressions other than those which may result from the evidence of truth. You will, no doubt, at the same time, have collected from the general scope of them, that they proceed from a source not unfriendly to the new Constitution. Yes, my countrymen, I own to you that, after having given it an attentive consideration, I am clearly of opinion it is your interest to adopt it. I am convinced that this is the safest course for your liberty, your dignity, and your happiness. I affect not reserves which I do not feel. I will not amuse you with an appearance of deliberation when I have decided. I frankly acknowledge to you my convictions, and I will freely lay before you the reasons on which they are founded. The consciousness of good intentions disdains ambiguity. I shall not, however, multiply professions on this head. My motives must remain in the depository of my own breast. My arguments will be open to all, and may be judged of by all. They shall at least be offered in a spirit which will not disgrace the cause of truth. It may be convenient to divide the subject into four heads: the utility of the UNION to your political prosperity, the insufficiency of the present Confederation to preserve that Union, the necessity of a government at least equally energetic with the one proposed to the attainment of this object, and the conformity of the proposed Constitution to the true principles of republican government. Whilst some may cavil, while we deliberate upon these great national objects, there is enough evidence to guide our inquiry.`
    },
    fed_10: {
      label: "Federalist No. 10 (James Madison)",
      context: "Federalist",
      trueStatus: "Undisputed James Madison (Note 'on', 'whilst', 'consequently', semicolon cadence)",
      text: `Among the numerous advantages promised by a well constructed Union, none deserves to be more accurately developed than its tendency to break and control the violence of faction. The friend of popular governments never finds himself so much alarmed for their character and fate, as when he contemplates their propensity to this dangerous vice. He will not fail, therefore, to set a due value on any plan which, without violating the principles to which he is attached, provides a proper cure for it. The instability, injustice, and confusion introduced into the public councils, have, in truth, been the mortal diseases under which popular governments have everywhere perished; as they continue to be the favorite and fruitful topics from which the adversaries to liberty derive their most specious declamations. The valuable improvements made by the American constitutions on the popular models, both ancient and modern, cannot certainly be too much admired; but it would be an unwarrantable partiality, to contend that they have as effectually obviated the danger on this side, as was wished and expected. By a faction, I understand a number of citizens, whether amounting to a majority or a minority of the whole, who are united and actuated by some common impulse of passion, or of interest, adversed to the rights of other citizens, or to the permanent and aggregate interests of the community. There are two methods of curing the mischiefs of faction: the one, by removing its causes; the other, by controlling its effects. There are again two methods of removing the causes of faction: the one, by destroying the liberty which is essential to its existence; the other, by giving to every citizen the same opinions, the same passions, and the same interests. It could never be more truly said than of the first remedy, that it was worse than the disease. Liberty is to faction what air is to fire, an aliment without which it instantly expires. But it could not be less folly to abolish liberty, which is essential to political life, because it nourishes faction, than it would be to wish the annihilation of air, which is essential to animal life, because it imparts to fire its destructive agency. The second expedient is as impracticable as the first would be unwise. As long as the reason of man continues fallible, and he is at liberty to exercise it, different opinions will be formed. As long as the connection subsists between his reason and his self-love, his opinions and his passions will have a reciprocal influence on each other; and the former will be objects to which the latter will attach themselves. The diversity in the faculties of men, from which the rights of property originate, is not less an insuperable obstacle to a uniformity of interests. The protection of these faculties is the first object of government. From the protection of different and unequal faculties of acquiring property, the possession of different degrees and kinds of property immediately results; and from the influence of these on the sentiments and views of the respective proprietors, ensues a division of the society into different interests and parties. The latent causes of faction are thus sown in the nature of man; and we see them everywhere brought into different degrees of activity, according to the different circumstances of civil society. A zeal for different opinions concerning religion, concerning government, and many other points, as well of speculation as of practice; an attachment to different leaders ambitiously contending for pre-eminence and power; or to persons of other descriptions whose fortunes have been interesting to the human passions, have, in turn, divided mankind into parties, inflamed them with mutual animosity, and rendered them much more disposed to vex and oppress each other than to co-operate for their common good. So strong is this propensity of mankind to fall into mutual animosities, that where no substantial occasion presents itself, the most frivolous and fanciful distinctions have been sufficient to kindle their unfriendly passions and excite their most violent conflicts. But the most common and durable source of factions has been the various and unequal distribution of property. Those who hold and those who are without property have ever formed distinct interests in society. Those who are creditors, and those who are debtors, fall under a like discrimination. A landed interest, a manufacturing interest, a mercantile interest, a moneyed interest, with many lesser interests, grow up of necessity in civilized nations, and divide them into different classes, actuated by different sentiments and views. The regulation of these various and interfering interests forms the principal task of modern legislation, and involves the spirit of party and faction in the necessary and ordinary operations of the government. No man is allowed to be a judge in his own cause, because his interest would certainly bias his judgment, and, not improbably, corrupt his integrity. With equal, nay with greater reason, a body of men are unfit to be both judges and parties at the same time; yet what are many of the most important acts of legislation, but so many judicial determinations, not indeed concerning the rights of single persons, but concerning the rights of large bodies of citizens? And what are the different classes of legislators but advocates and parties to the causes which they determine? Is a law proposed concerning private debts? It is a question to which the creditors are parties on one side and the debtors on the other. Justice ought to hold the balance between them. Yet the parties are, and must be, themselves the judges; and the most numerous party, or, in other words, the most powerful faction must be expected to prevail. Shall domestic manufactures be encouraged, and in what degree, by restrictions on foreign manufactures? are questions which would be differently decided by the landed and the manufacturing classes, and probably by neither with a sole regard to justice and the public good. The apportionment of taxes on the various descriptions of property is an act which seems to require the most exact impartiality; yet there is, perhaps, no legislative act in which greater opportunity and temptation are given to a predominant party to trample on the rules of justice. Every shilling with which they overburden the inferior number, is a shilling saved to their own pockets. It is in vain to say that enlightened statesmen will be able to adjust these clashing interests, and render them all subservient to the public good. Enlightened statesmen will not always be at the helm. Nor, in many cases, can such an adjustment be made at all without taking into view indirect and remote considerations, which will rarely prevail over the immediate interest which one party may find in disregarding the rights of another or the good of the whole. The inference to which we are brought is, that the causes of faction cannot be removed, and that relief is only to be sought in the means of controlling its effects. If a faction consists of less than a majority, relief is supplied by the republican principle, which enables the majority to defeat its sinister views by regular vote. It may clog the administration, it may convulse the society; but it will be unable to execute and mask its violence under the forms of the Constitution. When a majority is included in a faction, the form of popular government, on the other hand, enables it to sacrifice to its ruling passion or interest both the public good and the rights of other citizens. To secure the public good and private rights against the danger of such a faction, and at the same time to preserve the spirit and the form of popular government, is then the great object to which our inquiries are directed. Let me add that it is the great desideratum by which this form of government can be rescued from the opprobrium under which it has so long labored, and be recommended to the esteem and adoption of mankind. By what means is this object attainable? Evidently by one of two only. Either the existence of the same passion or interest in a majority at the same time must be prevented, or the majority, having such coexistent passion or interest, must be rendered, by their number and local situation, unable to concert and carry into effect schemes of oppression. If the impulse and the opportunity be suffered to coincide, we well know that neither moral nor religious motives can be relied on as an adequate control. They are not found to be such on the injustice and violence of individuals, and lose their efficacy in proportion to the number combined together, that is, in proportion as their efficacy becomes needful. From this view of the subject it may be concluded that a pure democracy, by which I mean a society consisting of a small number of citizens, who assemble and administer the government in person, can admit of no cure for the mischiefs of faction. A common passion or interest will, in almost every case, be felt by a majority of the whole; a communication and concert result from the form of government itself; and there is nothing to check the inducements to sacrifice the weaker party or an obnoxious individual. Hence it is that such democracies have ever been spectacles of turbulence and contention; have ever been found incompatible with personal security or the rights of property; and have in general been as short in their lives as they have been violent in their deaths. Theoretic politicians, who have patronized this species of government, have erroneously supposed that by reducing mankind to a perfect equality in their political rights, they would, at the same time, be perfectly equalized and assimilated in their possessions, their opinions, and their passions. A republic, by which I mean a government in which the scheme of representation takes place, opens a different prospect, and promises the cure for which we are seeking. Let us examine the points in which it varies from pure democracy, and we shall comprehend both the nature of the cure and the efficacy which it must derive from the Union. The two great points of difference between a democracy and a republic are: first, the delegation of the government, in the latter, to a small number of citizens elected by the rest; secondly, the greater number of citizens, and greater sphere of country, over which the latter may be extended. The effect of the first difference is, on the one hand, to refine and enlarge the public views, by passing them through the medium of a chosen body of citizens, whose wisdom may best discern the true interest of their country, and whose patriotism and love of justice will be least likely to sacrifice it to temporary or partial considerations. Under such a regulation, it may well happen that the public voice, pronounced by the representatives of the people, will be more consonant to the public good than if pronounced by the people themselves, convened for the purpose. On the other hand, the effect may be inverted. Men of factious tempers, of local prejudices, or of sinister designs, may, by intrigue, by corruption, or by other means, first obtain the suffrages, and then betray the interests, of the people. The question resulting is, whether small or extensive republics are more favorable to the election of proper guardians of the public weal; and it is clearly decided in favor of the latter by two obvious considerations: In the first place, it is to be remarked that, however small the republic may be, the representatives must be raised to a certain number, in order to guard against the cabals of a few; and that, however large it may be, they must be limited to a certain number, in order to guard against the confusion of a multitude. Hence, the number of representatives in the two cases not being in proportion to that of the two constituents, and being proportionally greater in the small republic, it follows that, if the proportion of fit characters be not less in the large than in the small republic, the former will present a greater option, and consequently a greater probability of a fit choice. In the next place, as each representative will be chosen by a greater number of citizens in the large than in the small republic, it will be more difficult for unworthy candidates to practice with success the vicious arts by which elections are too often carried; and the suffrages of the people being more free, will be more likely to centre in men who possess the most attractive merit and the most diffusive and established characters. It must be confessed that in this, as in most other cases, there is a mean, on both sides of which inconveniences will be found to lie. By enlarging too much the number of electors, you render the representative too little acquainted with all their local circumstances and lesser interests; as by reducing it too much, you render him unduly attached to these, and too little fit to comprehend and pursue great and national objects. The federal Constitution forms a happy combination in this respect; the great and aggregate interests being referred to the national, the local and particular to the State legislatures. The other point of difference is, the greater number of citizens and extent of territory which may be brought within the compass of republican than of democratic government; and it is this circumstance principally which renders factious combinations less to be dreaded in the former than in the latter. The smaller the society, the fewer probably will be the distinct parties and interests composing it; the fewer the distinct parties and interests, the more frequently will a majority be found of the same party; and the smaller the number of individuals composing a majority, and the smaller the compass within which they are placed, the more easily will they concert and execute their plans of oppression. Extend the sphere, and you take in a greater variety of parties and interests; you make it less probable that a majority of the whole will have a common motive to invade the rights of other citizens; or if such a common motive exists, it will be more difficult for all who feel it to discover their own strength, and to act in unison with each other. Besides other impediments, it may be remarked that, where there is a consciousness of unjust or dishonorable purposes, communication is always checked by distrust in proportion to the number whose concurrence is necessary. Hence, it clearly appears, that the same advantage which a republic has over a democracy, in controlling the effects of faction, is enjoyed by a large over a small republic,—is enjoyed by the Union over the States composing it. Whilst the mocks of faction may kindle a flame within their particular States, consequently they will be unable to spread a general conflagration through the other States.`
    },
    doyle_lost_world: {
      label: "The Lost World (Sci-Fi by Arthur Conan Doyle)",
      context: "Literature",
      trueStatus: "Held-out Science-Fiction Novel by Sir Arthur Conan Doyle (100% attributed to Doyle over H. G. Wells)",
      text: `We slowly and cautiously set forth upon our perilous journey across the plateau. Every step was taken with the utmost circumspection, for we knew well that at any instant we might stumble upon the lurking place of some prehistoric monster. Challenger walked in front, his great black beard bristling with excitement, while Summerlee trudged behind him in a state of continual and querulous protest. Lord John Roxton, with his elephant gun poised ready in the hollow of his arm, kept a vigilant eye upon the thickets of giant fern that bordered our track. "Look here, young fellah," he whispered to me as we paused beside a marshy pool, "did you ever see such tracks as those in all your born days?" Assuredly I never had. Deeply impressed in the soft mud were huge three-toed footprints, each of them large enough to hold a wash-basin, and arranged in a long stride that bespoke a creature of stupendous bulk and agility. "Iguanodon," said Challenger pompously, waving his thick hand toward the glade. "You will observe, my dear Summerlee, that the plantar surface exhibits the exact contour which I had predicated in my monograph upon the Wealden footprints." Summerlee peered through his spectacles with an air of obstinate skepticism, though even he was forced to admit that no existing animal of the Amazonian basin could have left such a spoor. For three hours we pushed onward through the tangled brake, our nerves strung to the highest pitch, listening to the strange, guttural croakings that rose from the cane-swamps upon our left.`
    },
    bronte_villette: {
      label: "Villette (Charlotte Brontë)",
      context: "Literature",
      trueStatus: "Held-out Novel by Charlotte Brontë (Note signature semicolon `;` and colon `:` density)",
      text: `My godmother lived in a handsome house in the clean and ancient town of Bretton. Her husband's family had been residents there for generations, and bore, indeed, the name of their birthplace—Bretton of Bretton: whether by coincidence, or because some remote ancestor had been a personage of sufficient importance to leave hisronymic to his neighborhood, I know not. When I was a girl I went to Bretton about twice a year, and well I liked the visit. The house and its inmates specially suited me: the large peaceful rooms, the well-arranged furniture, the clear wide windows, the quiet routine of existence—all these things pleased my temper and calmed my spirits. Time always flowed smoothly for me at my godmother's side; not with tumultuous rush, nor with painful stagnation; but with a tranquil, measured lapse, like a clear stream gliding through a level meadow: no rapids agitated its current; no rocks fretted its bed; no dark overhanging trees cast gloom upon its margin. Why should I have wished to change such a scene? Yet change was approaching: the shadow of an unknown trouble already lay across the threshold; and before the autumn leaves had fallen, that quiet household was destined to know both anxiety and sorrow.`
    }
  },

  referenceCurves: {
    "Hamilton (Federalist)": [0.025, 0.211, 0.209, 0.141, 0.095, 0.075, 0.068, 0.056, 0.046, 0.036, 0.019, 0.011, 0.005, 0.002, 0.001],
    "Madison (Federalist)":  [0.024, 0.206, 0.206, 0.135, 0.092, 0.074, 0.072, 0.058, 0.050, 0.041, 0.021, 0.012, 0.006, 0.002, 0.001],
    "Charlotte Brontë":      [0.041, 0.172, 0.218, 0.194, 0.118, 0.088, 0.068, 0.044, 0.028, 0.016, 0.007, 0.004, 0.001, 0.001, 0.000],
    "Arthur Conan Doyle":    [0.044, 0.184, 0.229, 0.197, 0.114, 0.082, 0.062, 0.040, 0.024, 0.013, 0.006, 0.003, 0.001, 0.001, 0.000],
    "H. G. Wells":           [0.042, 0.181, 0.224, 0.188, 0.116, 0.086, 0.066, 0.043, 0.027, 0.015, 0.007, 0.003, 0.001, 0.001, 0.000]
  },

  disputedFederalist: [
    { num: 49, words: 1646, title: "Appealing to the People Through a Convention", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.812, bM: 0.731, cH: 1.326, cM: 0.674, mwM: 0.910, giM: 0.90, upon: 0.00, on: 8.51, whilst: 0.61, while: 0.00, enough: 0.00, conseq: 0.00 },
    { num: 50, words: 1101, title: "Periodical Appeals to the People Considered", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 1.139, bM: 1.102, cH: 1.278, cM: 0.722, mwM: 0.982, giM: 0.94, upon: 0.00, on: 9.08, whilst: 0.00, while: 0.00, enough: 0.00, conseq: 0.91 },
    { num: 51, words: 1914, title: "Checks and Balances Between Departments", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.986, bM: 0.861, cH: 1.394, cM: 0.606, mwM: 1.000, giM: 0.88, upon: 0.00, on: 8.88, whilst: 1.04, while: 0.00, enough: 0.00, conseq: 0.52 },
    { num: 52, words: 1844, title: "The House of Representatives & Qualifications", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.851, bM: 0.775, cH: 1.363, cM: 0.637, mwM: 0.950, giM: 0.60, upon: 0.00, on: 8.13, whilst: 0.00, while: 0.00, enough: 0.00, conseq: 0.54 },
    { num: 53, words: 2164, title: "Biennial Elections of the House", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.808, bM: 0.722, cH: 1.328, cM: 0.672, mwM: 0.963, giM: 0.90, upon: 0.00, on: 6.93, whilst: 0.46, while: 0.00, enough: 0.00, conseq: 0.00 },
    { num: 54, words: 1996, title: "Apportionment of Members Among the States", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.874, bM: 0.804, cH: 1.303, cM: 0.697, mwM: 0.974, giM: 0.84, upon: 1.00, on: 6.01, whilst: 0.00, while: 0.00, enough: 0.00, conseq: 0.50 },
    { num: 55, words: 2043, title: "The Total Number of the House (Historical Split Case)", consensus: "Madison", vote: "6/11 (54.5%)", bct: "30/48 (62.5%)", bH: 0.828, bM: 0.851, cH: 1.134, cM: 0.866, mwM: 0.383, giM: 0.64, upon: 0.00, on: 4.41, whilst: 0.00, while: 0.00, enough: 0.00, conseq: 0.00 },
    { num: 56, words: 1650, title: "Knowledge of Local Circumstances in the House", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.979, bM: 0.964, cH: 1.217, cM: 0.783, mwM: 0.787, giM: 0.83, upon: 0.00, on: 4.24, whilst: 0.61, while: 0.00, enough: 0.00, conseq: 0.00 },
    { num: 57, words: 2206, title: "Alleged Tendency to Elevate the Few", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.784, bM: 0.731, cH: 1.268, cM: 0.732, mwM: 0.972, giM: 0.74, upon: 0.00, on: 9.07, whilst: 1.36, while: 0.00, enough: 0.00, conseq: 0.00 },
    { num: 58, words: 2080, title: "Augmentation of Members as Population Increases", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.755, bM: 0.685, cH: 1.308, cM: 0.692, mwM: 0.948, giM: 0.86, upon: 0.48, on: 8.65, whilst: 0.48, while: 0.00, enough: 0.00, conseq: 0.00 },
    { num: 62, words: 2387, title: "The Senate: Qualifications & Equal Representation", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.745, bM: 0.707, cH: 1.227, cM: 0.773, mwM: 0.956, giM: 0.86, upon: 0.00, on: 5.45, whilst: 0.00, while: 0.00, enough: 0.00, conseq: 0.84 },
    { num: 63, words: 3039, title: "The Senate Continued: National Character", consensus: "Madison", vote: "11/11 (100%)", bct: "48/48 (100%)", bH: 0.716, bM: 0.654, cH: 1.309, cM: 0.691, mwM: 0.879, giM: 0.90, upon: 0.00, on: 7.57, whilst: 0.33, while: 0.00, enough: 0.00, conseq: 0.00 }
  ],

  unmaskingCurves: {
    rounds: [1, 2, 3, 4, 5, 6, 7, 8],
    disputed_vs_hamilton: [1.000, 1.000, 0.950, 0.900, 0.833, 0.800, 0.750, 0.717],
    disputed_vs_madison:  [0.783, 0.700, 0.650, 0.567, 0.533, 0.433, 0.467, 0.417],
    hamilton_split_same:  [0.883, 0.800, 0.733, 0.717, 0.700, 0.600, 0.567, 0.500]
  },

  limitsGuide: [
    {
      num: "Limit 01",
      title: "The Sample Size Floor (Minimum Text Length)",
      summary: "Stylometric word frequencies are governed by Poisson/multinomial sampling noise. In short texts (<500–1,000 words, such as emails, ransom notes, or short letters), 70%+ of the top 150 function words occur zero or one time, producing wild z-score fluctuations.",
      mitigation: "Rule of Thumb (Eder, 2015): Require ≥2,500 words (and ideally ≥5,000 words) per text for reliable Burrows/Cosine Delta attribution. Below 1,000 words, always report wide confidence intervals and combine character n-grams with open-set verification."
    },
    {
      num: "Limit 02",
      title: "The Closed-Set Fallacy ('Forced Lineup Error')",
      summary: "Closed-set distance methods (Burrows's Delta, Chi-Squared, SVM, Logistic Regression) will ALWAYS output whichever candidate in your training folder has the smallest distance—even if the true author is a third party who isn't in your dataset at all.",
      mitigation: "Never use closed-set Delta alone in an open-world investigation. Always run Koppel & Winter's General Imposters (GI) verification or Koppel's Unmasking with a diverse lineup of distractor ('imposter') authors from the same era and genre."
    },
    {
      num: "Limit 03",
      title: "Genre, Register & Narrative Perspective Confounding",
      summary: "An author writing a first-person romantic novel uses very different pronoun rates ('I', 'my', 'her') and sentence lengths than the same author writing third-person history or political journalism. Comparing across mismatched genres can cause texts to cluster by genre rather than author.",
      mitigation: "Match reference corpora by genre and era. In literary attribution, enable pronoun removal (`remove_pronouns=True`) and Hoover document-frequency culling (`culling_ratio=0.2` to `0.5`) to eliminate character names and narrative-person bias."
    },
    {
      num: "Limit 04",
      title: "Collaborative Authorship, Ghostwriting & Editorial Hands",
      summary: "Historical and modern texts are rarely 100% single-author purity: secretaries, copy-editors, translators, and co-authors (e.g., Madison and Hamilton in Federalist Nos. 18–20, or disputed Federalist No. 55) create blended stylistic profiles that mislead whole-document classifiers.",
      mitigation: "Use Eder's Rolling Stylometry (`RollingStylometryAnalyzer`) to inspect sequential 500–1,000 word windows across the document and check whether the stylistic signal is uniform or shifts mid-text."
    },
    {
      num: "Limit 05",
      title: "Chronological Drift ('Late Style')",
      summary: "An author's linguistic fingerprint is not frozen at birth: vocabulary diversity, clause complexity, and function-word preferences drift gradually over decades (e.g., Henry James's early vs. late novels, or changes associated with aging and cognitive decline).",
      mitigation: "Include training texts that bracket the chronological period of the disputed work, and test whether Delta distances correlate with publication date."
    },
    {
      num: "Limit 06",
      title: "Adversarial Stylometry & LLM Paraphrasing",
      summary: "As shown by Brennan, Afroz & Greenstadt (2012), when an author consciously tries to obfuscate their style (reducing sentence length, swapping function words) or imitates another writer—or passes text through an LLM—standard Delta accuracy drops sharply.",
      mitigation: "Combine multiple orthogonal feature families (function words + punctuation cadence + morphosyntactic POS transitions + character n-grams) and examine Unmasking degradation curves to detect artificial style smoothing."
    }
  ]
};
