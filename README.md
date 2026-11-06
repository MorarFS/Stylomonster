# Stylomonster (`stylometry`) — Computational Stylometry & Authorship Attribution

- **GitHub Repository**: [https://github.com/MorarFS/Stylomonster](https://github.com/MorarFS/Stylomonster)
- **Live Interactive Website & Lab (GitHub Pages)**: [https://morarfs.github.io/Stylomonster/](https://morarfs.github.io/Stylomonster/)

A complete, modular Python package, CLI, and interactive educational web application that consolidates **17 classical and modern stylometric methods**, feature extractors, statistical distance measures, authorship verification algorithms, sequential/collaborative analysis tools, and benchmark corpus loaders into a single library.

## Package & Repository Structure

```text
Stylomonster/
├── pyproject.toml
├── README.md
├── RESOURCES.md                         # Complete mathematical, algorithmic & bibliographic catalog
├── src/stylometry/
│   ├── __init__.py                      # Public API exports (v1.1.0)
│   ├── resources.py                     # 17-method catalog, Mosteller-Wallace 30 markers, 180 function words
│   ├── preprocessing.py                 # Normalization, tokenization, MFW selection, Hoover culling, chunking
│   ├── lexical.py                       # TTR, Guiraud's R, Herdan's C, MATTR, Yule's K, Simpson's D,
│   │                                    # Honoré's R, Sichel's S, Brunét's W, Mendenhall's Curves (1887)
│   ├── syntactic.py                     # Sentence rhythm, punctuation fingerprints (; : , — ? ! () ""),
│   │                                    # 5 readability indices, function word rates, Character 3/4-grams
│   ├── pos_stylometry.py                # Morphosyntactic / POS Tag 1/2/3-gram transition stylometry
│   ├── delta.py                         # Burrows, Cosine (Würzburg), Quadratic (Argamon), Eder,
│   │                                    # Canberra & MinMax Delta + per-feature attribution explainability
│   ├── chi_square.py                    # Kilgarriff's (2001) Chi-Squared (χ²) corpus homogeneity distance
│   ├── imposters.py                     # Koppel & Winter's (2014) General Imposters (GI) verification
│   ├── zeta.py                          # Burrows/Craig Zeta (ζ) distinctive words & Koppel's Unmasking (2004)
│   ├── rolling.py                       # Eder's (2016) Rolling Stylometry for sequential / co-authored texts
│   ├── consensus_tree.py                # Eder's (2012, 2016) Bootstrap Consensus Tree (BCT) multi-MFW stability
│   ├── classifiers.py                   # Mosteller-Wallace Discriminant, Composite ML, PCA/Ward Clustering,
│   │                                    # and 12-Method Consensus Ensemble Attributor
│   ├── corpus.py                        # Parsers for The Federalist Papers (85 essays) & 19th-C. Novels (15 books)
│   └── cli.py                           # `stylometry` command-line interface (`resources`, `profile`, `attribute`)
├── scripts/
│   ├── 01_download_corpora.py           # Downloads & parses Federalist (85) and Literary (45 segments) corpora
│   ├── 02_extract_features.py           # Exports full CSV feature tables (87 columns per document)
│   ├── 03_test_authorship.py            # Runs 12-method cross-validation & disputed authorship benchmarks
│   ├── 04_rolling_zeta_unmasking.py     # Runs Zeta markers, Rolling Stylometry, Unmasking curves & 48-model BCT
│   ├── 05_attribute_custom_text.py      # Standalone forensic attribution script for any custom text/folder
│   ├── run_experiments.py               # Core benchmark runner invoked by 03_test_authorship.py
│   └── run_all.sh                       # Master end-to-end execution script (pytest + scripts 01..05)
├── docs/                                # Interactive Educational Website & Live Stylometry Lab (GitHub Pages)
│   ├── index.html                       # Semantic HTML5 single-page application
│   ├── styles.css                       # Editorial Vanilla CSS design system (Light & Dark modes)
│   ├── data.js                          # 17-method pedagogical store, presets & benchmark results
│   └── app.js                           # Real-time client-side stylometric engine & Canvas charts
├── data/
│   ├── raw/                             # Project Gutenberg source texts (`pg1404.txt` + 15 novels)
│   └── processed/                       # Clean JSON corpora, CSV feature matrices & JSON benchmark reports
└── tests/
    └── test_stylometry.py               # Comprehensive pytest unit & integration test suite
```

## Interactive Web Application (`docs/` & GitHub Pages)

The repository includes a zero-dependency interactive web application in [`docs/index.html`](docs/index.html) published via GitHub Pages at **[https://morarfs.github.io/Stylomonster/](https://morarfs.github.io/Stylomonster/)**:
- **17-Method Interactive Compendium**: Filterable dossier for every stylometric method with conceptual intuition, exact mathematical formulas, best-practice parameters, failure modes/limits, and copyable Python snippets.
- **Live Browser Stylometry Sandbox**: Paste any text or load historical presets to compute real-time MATTR, Yule's $K$, Honoré's $R$, semicolon/colon density, function-word marker rates (`upon` vs. `on`, `while` vs. `whilst`), and an interactive Mendenhall word-length curve `<canvas>`.
- **Empirical Benchmark & Unmasking Explorer**: Inspect all 12 disputed *Federalist Papers*, Koppel's Unmasking degradation curves, and the 19th-Century Novelists attribution benchmark.
- **Educational Guide to Use & Limits**: Interactive **Sample-Size Reliability Simulator** (100 to 10,000 words) and 6 fundamental epistemological limits of stylometry.

```bash
# Serve the website locally on http://localhost:8080
python3 -m http.server 8080 --directory docs
```

## Quickstart

### 1. Run the Complete Pipeline & Test Suite
```bash
./scripts/run_all.sh
```

### 2. Run Individual Scripts
```bash
# 1. Download & parse corpora to JSON
.venv/bin/python scripts/01_download_corpora.py

# 2. Extract 87-column stylometric feature CSVs
.venv/bin/python scripts/02_extract_features.py

# 3. Run 12-method cross-validation & disputed authorship benchmark
.venv/bin/python scripts/03_test_authorship.py

# 4. Run Craig's Zeta, Rolling Stylometry, Unmasking & Bootstrap Consensus
.venv/bin/python scripts/04_rolling_zeta_unmasking.py

# 5. Run a forensic attribution report on any paper or custom text file
.venv/bin/python scripts/05_attribute_custom_text.py --corpus federalist --paper 51
.venv/bin/python scripts/05_attribute_custom_text.py --corpus literature --query data/raw/novels/doyle_lost_world.txt
```

### 3. Command-Line Interface (`stylometry`)
```bash
# View the complete catalog of 17 stylometric methods, word lists, and corpora
.venv/bin/stylometry resources

# Profile the lexical, syntactic, and Mendenhall fingerprint of any text file
.venv/bin/stylometry profile path/to/document.txt

# Attribute an unknown text against a directory of known author `.txt` files
.venv/bin/stylometry attribute --train-dir path/to/known_authors --query path/to/unknown.txt
```
