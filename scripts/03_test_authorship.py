#!/usr/bin/env python3
"""
Script 03: Full 12-Method Authorship Attribution & Cross-Validation Benchmark
-----------------------------------------------------------------------------
Wrapper invoking the complete benchmark suite (`run_experiments.py`) across
all 12 stylometric methods on both The Federalist Papers (85 essays) and the
19th-Century Classic Literature Corpus (15 novels / 45 segments).
"""

from run_experiments import main

if __name__ == "__main__":
    main()
