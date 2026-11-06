# Comprehensive Resources for Stylometric Analysis of Texts

This catalog compiles the mathematical methods, canonical feature sets, open-source software packages, benchmark corpora, and foundational literature for computational stylometry and authorship attribution. All methods and word lists documented below are implemented in the [`stylometry`](file:///home/antigravity/stylometry_toolkit/src/stylometry/__init__.py) Python package.

---

## 1. Taxonomy of Stylometric Features

Computational stylometry measures unconscious linguistic habits that remain invariant across topics and genres:

| Feature Family | Specific Markers / Indices | Stylometric Signal & Rationale | Module |
| :--- | :--- | :--- | :--- |
| **Function Words & High-Frequency MFWs** | Top $M \in [50, 500]$ Most Frequent Words; 180 closed-class English function words; 30 Mosteller–Wallace discriminant words (*upon*, *on*, *while*, *whilst*, *enough*, *consequently*, etc.) | Topic-independent grammatical glue (prepositions, conjunctions, determiners, auxiliary verbs, discourse particles) governed by unconscious habit rather than subject matter. | [`stylometry.resources`](file:///home/antigravity/stylometry_toolkit/src/stylometry/resources.py), [`stylometry.delta`](file:///home/antigravity/stylometry_toolkit/src/stylometry/delta.py) |
| **Lexical Richness & Vocabulary Diversity** | Type-Token Ratio (TTR), Guiraud's $R$, Herdan's $\log\text{TTR}$, Moving-Average TTR ($\text{MATTR}$), Yule's $K$ Characteristic, Simpson's $D$, Honoré's $R$, Sichel's $S$, Brunét's $W$, Hapax Legomena ratio ($V_1/V$) | Captures vocabulary size, word repetition rate, and reliance on once-only words (*hapax legomena*) while controlling for document length. | [`stylometry.lexical`](file:///home/antigravity/stylometry_toolkit/src/stylometry/lexical.py) |
| **Word-Length Composition Curves** | Mendenhall's (1887) Characteristic Curves of Composition $P(|w| = k)$ for $k \in \{1,\dots,16\}$ | Captures Anglo-Saxon vs. Latinate/Romance lexical preference through word-length histograms. | [`stylometry.lexical`](file:///home/antigravity/stylometry_toolkit/src/stylometry/lexical.py) |
| **Syntactic Rhythm & Punctuation Fingerprints** | Sentence length distribution (mean, SD, median, IQR, short $\le 12$ and long $\ge 35$ word sentence ratios); rates per 1,000 tokens of `;`, `:`, `,`, `—`, `?`, `!`, `()`, `""`; Readability indices (Flesch, Kincaid, Coleman–Liau, ARI, Gunning Fog) | Punctuation marks—especially semicolons, colons, and em-dashes—act as direct proxies for clause parataxis/hypotaxis and authorial cadence. | [`stylometry.syntactic`](file:///home/antigravity/stylometry_toolkit/src/stylometry/syntactic.py) |
| **Character $n$-Gram Profiles (CNG)** | Top-500 character 3-grams and 4-grams (including word boundaries and punctuation affixes) | Captures sub-word morphology, inflectional suffixes (`-ing`, `-tion`, `-ed`), prefixes, spelling variants, and punctuation spacing without language-specific parsers. | [`stylometry.syntactic`](file:///home/antigravity/stylometry_toolkit/src/stylometry/syntactic.py) |

---

## 2. Core Mathematical & Statistical Attribution Methods

### 2.1 Burrows's Classic Delta ($\Delta_B$, 2002)
For the $M$ Most Frequent Words, let $f_i(D)$ be the relative frequency of word $i$ in document $D$, and let $\mu_i, \sigma_i$ be the mean and standard deviation of word $i$ across the reference corpus. The standardized $z$-score is:
$$z_i(D) = \frac{f_i(D) - \mu_i}{\sigma_i}$$
Burrows's Classic Delta is the mean absolute ($L_1$ / Manhattan) distance between the $z$-score profiles of documents $D_1$ and $D_2$:
$$\Delta_B(D_1, D_2) = \frac{1}{M} \sum_{i=1}^{M} \left| z_i(D_1) - z_i(D_2) \right|$$

### 2.2 Cosine Delta / Würzburg Delta ($\Delta_{\cos}$, Smith & Aldridge 2011; Evert et al. 2017)
Normalizes the $z$-score vectors onto the unit hypersphere before computing angular distance:
$$\Delta_{\cos}(D_1, D_2) = 1 - \frac{\mathbf{z}(D_1) \cdot \mathbf{z}(D_2)}{\|\mathbf{z}(D_1)\|_2 \, \|\mathbf{z}(D_2)\|_2}$$
Demonstrated by Evert et al. (2017) to be the most robust Delta variant across large $M$ values because $L_2$ normalization prevents single outlier words from dominating the distance.

### 2.3 Argamon's Quadratic Delta ($\Delta_Q$, 2008) & Eder's Delta ($\Delta_E$, 2016)
- **Quadratic Delta ($\Delta_Q$)**: Root-mean-square Euclidean ($L_2$) distance over $z$-scores:
  $$\Delta_Q(D_1, D_2) = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \left( z_i(D_1) - z_i(D_2) \right)^2}$$
- **Eder's Delta ($\Delta_E$)**: Applies an inverse rank weight $w_i = \frac{M - i + 1}{M}$ so higher-ranked function words carry greater weight than lower-ranked words:
  $$\Delta_E(D_1, D_2) = \frac{\sum_{i=1}^{M} w_i \left| z_i(D_1) - z_i(D_2) \right|}{\sum_{i=1}^{M} w_i}$$

### 2.4 Kilgarriff's Chi-Squared ($\chi^2$, 2001)
Given raw token counts $C_A(w)$ and $C_B(w)$ in candidate subcorpus $A$ (size $N_A$) and test document $B$ (size $N_B$) over the top $M$ words:
$$E_{w,A} = (C_A(w) + C_B(w)) \frac{N_A}{N_A + N_B}, \qquad E_{w,B} = (C_A(w) + C_B(w)) \frac{N_B}{N_A + N_B}$$
$$\chi^2(A, B) = \sum_{w \in \text{MFW}} \left[ \frac{(C_A(w) - E_{w,A})^2}{E_{w,A}} + \frac{(C_B(w) - E_{w,B})^2}{E_{w,B}} \right]$$

### 2.5 Mosteller & Wallace (1964) Function-Word Discriminant Model
Focuses on 30 high-discriminant non-contextual marker words (`upon`, `whilst`, `while`, `enough`, `there`, `on`, `to`, `by`, `of`, `from`, `in`, `also`, `an`, `both`, `though`, `although`, `commonly`, `consequently`, `considerable`, `according`, `apt`, `direction`, `innovation`, `language`, `vigor`, `kind`, `matter`, `particularly`, `probability`, `work`). For instance, in *The Federalist Papers*, Alexander Hamilton uses *upon* at $\sim 3.2$ per 1,000 words vs. James Madison's $\sim 0.16$ per 1,000 words, and Hamilton exclusively writes *while* whereas Madison prefers *whilst*.

### 2.6 Koppel & Winter's General Imposters (GI) Method (2014)
An open-set authorship verification algorithm. Across $K$ Monte Carlo iterations:
1. Randomly sample $50\%$ of the feature space $F_k \subset F$.
2. Randomly sample a subset of distractor ("imposter") documents $I_k$ from other authors.
3. Check whether disputed document $X$ is closer to candidate $C$ than to the closest imposter in $I_k$:
$$S(X, C) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}\!\left[ \min_{c \in C} d_{F_k}(X, c) < \min_{i \in I_k} d_{F_k}(X, i) \right]$$

---

## 3. External Software & Ecosystem Reference

| Package / Resource | Language | Authors / Maintainers | Focus & Capabilities |
| :--- | :--- | :--- | :--- |
| **`stylo`** | R | Maciej Eder, Jan Rybicki, Mike Kestemont (Computational Stylistics Group) | Standard Digital Humanities suite for Delta, PCA, Bootstrap Consensus Trees, Rolling Stylometry, and `imposters()`. |
| **`faststylometry`** | Python | Thomas Wood (Fast Data Science) | Burrows's Delta with logistic regression probability calibration. |
| **`StyloMetrix`** | Python | ZILiAT-NASK | spaCy-backed grammatical, syntactic, and lexical feature extraction. |
| **`pystyl`** | Python | Mike Kestemont et al. | Python stylometric preprocessing, vectorization, and clustering. |
| **PAN Shared Tasks** | Multi | Webis Group (CLEF) | Annual forensic stylometry benchmarks for authorship attribution, verification, and style breach detection. |

---

## 4. Seminal Bibliography

1. **Mendenhall, T. C. (1887).** *The Characteristic Curves of Composition.* Science, 9(214), 237–249.
2. **Yule, G. U. (1944).** *The Statistical Study of Literary Vocabulary.* Cambridge University Press.
3. **Mosteller, F., & Wallace, D. L. (1964).** *Inference and Disputed Authorship: The Federalist.* Addison-Wesley.
4. **Kilgarriff, A. (2001).** *Comparing Corpora.* International Journal of Corpus Linguistics, 6(1), 97–133.
5. **Burrows, J. F. (2002).** *'Delta': A Measure of Stylistic Difference and a Guide to Likely Authorship.* Literary and Linguistic Computing, 17(3), 267–287.
6. **Kešelj, V., Peng, F., Cercone, N., & Thomas, C. (2003).** *N-gram-based Author Profiles for Authorship Attribution.* PACLING '03, 255–264.
7. **Argamon, S. (2008).** *Interpreting Burrows's Delta: Geometric and Probabilistic Foundations.* Literary and Linguistic Computing, 23(2), 131–147.
8. **Stamatatos, E. (2009).** *A Survey of Modern Authorship Attribution Methods.* JASIST, 60(3), 538–556.
9. **Covington, M. A., & McFall, J. D. (2010).** *Cutting the Gordian Knot: The Moving-Average Type–Token Ratio (MATTR).* Journal of Quantitative Linguistics, 17(2), 94–100.
10. **Koppel, M., & Winter, Y. (2014).** *Determining If Two Documents Are Written by the Same Author.* JASIST, 65(1), 178–187.
11. **Eder, M., Rybicki, J., & Kestemont, M. (2016).** *Stylometry with R: A Package for Computational Text Analysis.* The R Journal, 8(1), 107–121.
12. **Evert, S., Proisl, T., Jannidis, F., Reger, I., Pielström, S., Schöch, C., & Vitt, T. (2017).** *Understanding and Explaining Delta Measures for Authorship Attribution.* Digital Scholarship in the Humanities, 32(suppl_2), ii4–ii16.
