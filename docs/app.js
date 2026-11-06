/* ==========================================================================
   Computational Stylometry & Authorship Attribution — Interactive Logic (app.js)
   ========================================================================== */

(function () {
  "use strict";

  const DATA = window.STYLOMETRY_DATA;
  let activeCategory = "All";
  let activeMethodId = DATA.methods[0].id;
  let activeFederalistNum = 51;
  let activeReferenceAuthor = "Madison (Federalist)";

  // -------------------------------------------------------------------------
  // 1. Theme Toggle
  // -------------------------------------------------------------------------
  function initTheme() {
    const btn = document.getElementById("theme-toggle-btn");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      btn.textContent = next === "dark" ? "Light Paper Mode" : "Dark Ink Mode";
      runLiveSandboxAnalysis();
      drawUnmaskingCanvas();
    });
  }

  // -------------------------------------------------------------------------
  // 2. Interactive 17-Method Compendium
  // -------------------------------------------------------------------------
  function renderCategoryFilters() {
    const categories = ["All", ...Array.from(new Set(DATA.methods.map((m) => m.category)))];
    const bar = document.getElementById("method-category-bar");
    if (!bar) return;
    bar.innerHTML = "";
    categories.forEach((cat) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.id = "filter-cat-" + cat.toLowerCase().replace(/[^a-z0-9]+/g, "-");
      btn.className = "filter-btn" + (cat === activeCategory ? " active" : "");
      const count = cat === "All" ? DATA.methods.length : DATA.methods.filter((m) => m.category === cat).length;
      btn.textContent = `${cat} (${count})`;
      btn.addEventListener("click", () => {
        activeCategory = cat;
        const filtered = getFilteredMethods();
        if (!filtered.some((m) => m.id === activeMethodId) && filtered.length > 0) {
          activeMethodId = filtered[0].id;
        }
        renderCategoryFilters();
        renderMethodsList();
        renderMethodDossier();
      });
      bar.appendChild(btn);
    });
  }

  function getFilteredMethods() {
    if (activeCategory === "All") return DATA.methods;
    return DATA.methods.filter((m) => m.category === activeCategory);
  }

  function renderMethodsList() {
    const listEl = document.getElementById("methods-list-container");
    if (!listEl) return;
    listEl.innerHTML = "";
    const filtered = getFilteredMethods();

    filtered.forEach((m) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.id = "select-method-" + m.id;
      btn.className = "method-item-btn" + (m.id === activeMethodId ? " active" : "");
      btn.innerHTML = `
        <div class="method-item-top">
          <span class="method-item-name">${escapeHtml(m.name)}</span>
          <span class="method-item-year">${m.year}</span>
        </div>
        <div class="method-item-cat">${escapeHtml(m.category)} • ${escapeHtml(m.authors)}</div>
      `;
      btn.addEventListener("click", () => {
        activeMethodId = m.id;
        renderMethodsList();
        renderMethodDossier();
      });
      listEl.appendChild(btn);
    });
  }

  function renderMethodDossier() {
    const container = document.getElementById("method-dossier-container");
    if (!container) return;
    const m = DATA.methods.find((x) => x.id === activeMethodId) || DATA.methods[0];

    const useItems = m.useCases.map((u) => `<li>${escapeHtml(u)}</li>`).join("");
    const limitItems = m.limits.map((l) => `<li>${escapeHtml(l)}</li>`).join("");

    container.innerHTML = `
      <div class="dossier-header">
        <div class="dossier-meta">
          <span class="badge badge-primary">Introduced ${m.year}</span>
          <span class="badge badge-secondary">${escapeHtml(m.category)}</span>
          <span class="badge">${escapeHtml(m.module)}</span>
        </div>
        <h3>${escapeHtml(m.name)}</h3>
        <p style="margin-top: 0.35rem; font-size: 0.9rem;"><strong>Seminal Reference:</strong> ${escapeHtml(m.authors)}</p>
      </div>

      <div>
        <h4 style="font-family: var(--font-sans); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 0.4rem;">Conceptual Intuition</h4>
        <p style="font-size: 0.95rem; color: var(--text-primary); max-width: 100%;">${escapeHtml(m.intuition)}</p>
      </div>

      <div class="formula-box">
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 0.3rem;">Mathematical Formulation</div>
        <div>${escapeHtml(m.formula)}</div>
      </div>

      <div class="dossier-columns">
        <div class="info-panel use-panel">
          <h4>✓ When &amp; How to Use</h4>
          <ul>${useItems}</ul>
        </div>
        <div class="info-panel limit-panel">
          <h4>⚠ Critical Limits &amp; Failure Modes</h4>
          <ul>${limitItems}</ul>
        </div>
      </div>

      <div class="code-block-wrap">
        <div class="code-block-header">
          <span>Python Package Usage (${escapeHtml(m.module)})</span>
          <button type="button" id="copy-code-btn" class="btn btn-sm" style="padding: 0.2rem 0.55rem; font-size: 0.74rem;">Copy Snippet</button>
        </div>
        <pre><code id="dossier-code-text">${escapeHtml(m.code)}</code></pre>
      </div>
    `;

    const copyBtn = document.getElementById("copy-code-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(m.code).then(() => {
          copyBtn.textContent = "Copied!";
          setTimeout(() => {
            copyBtn.textContent = "Copy Snippet";
          }, 1500);
        });
      });
    }
  }

  // -------------------------------------------------------------------------
  // 3. Live Client-Side Stylometric Engine & Sandbox
  // -------------------------------------------------------------------------
  function countSyllablesWord(w) {
    let clean = w.toLowerCase().replace(/[^a-z]/g, "");
    if (clean.length <= 3) return 1;
    if (clean.endsWith("e") && !clean.endsWith("le")) clean = clean.slice(0, -1);
    else if ((clean.endsWith("es") || clean.endsWith("ed")) && clean.length > 4) clean = clean.slice(0, -2);
    const matches = clean.match(/[aeiouy]+/g);
    return matches ? Math.max(1, matches.length) : 1;
  }

  function computeClientStylometrics(rawText) {
    const norm = rawText.replace(/\r\n/g, "\n").replace(/--+/g, "\u2014");
    const wordMatches = norm.match(/[a-zA-Z]+(?:'[a-zA-Z]+)?/g) || [];
    const tokens = wordMatches.map((w) => w.toLowerCase());
    const n = tokens.length;

    if (n === 0) {
      return null;
    }

    const counts = Object.create(null);
    let totalChars = 0;
    let totalSyllables = 0;
    const curve = new Array(15).fill(0);

    for (let i = 0; i < n; i++) {
      const w = tokens[i];
      counts[w] = (counts[w] || 0) + 1;
      totalChars += w.length;
      totalSyllables += countSyllablesWord(w);
      const bin = Math.min(15, Math.max(1, w.length)) - 1;
      curve[bin] += 1;
    }
    for (let k = 0; k < 15; k++) curve[k] /= n;

    const types = Object.keys(counts);
    const v = types.length;

    // Spectrum V_i
    const spectrum = Object.create(null);
    for (let i = 0; i < v; i++) {
      const f = counts[types[i]];
      spectrum[f] = (spectrum[f] || 0) + 1;
    }
    const v1 = spectrum[1] || 0;

    // MATTR-100
    const win = 100;
    let mattr = v / n;
    if (n > win) {
      let sumUnique = 0;
      let windowsCount = 0;
      for (let start = 0; start <= n - win; start += 25) {
        const sub = new Set(tokens.slice(start, start + win));
        sumUnique += sub.size / win;
        windowsCount++;
      }
      mattr = sumUnique / windowsCount;
    }

    // Yule's K
    let s2 = 0;
    for (const freqStr of Object.keys(spectrum)) {
      const freq = Number(freqStr);
      s2 += freq * freq * spectrum[freqStr];
    }
    const yulesK = (10000 * (s2 - n)) / (n * n);

    // Honoré's R
    const denom = 1 - v1 / v;
    const honoresR = denom > 1e-5 ? (100 * Math.log(n)) / denom : 100 * Math.log(n);

    // Sentences
    const rawSents = norm.split(/(?<=[.!?])\s+|\n\s*\n/).filter((s) => /[a-zA-Z]/.test(s));
    const nSents = Math.max(1, rawSents.length);
    const meanSentLen = n / nSents;

    // Punctuation per 1,000 words
    const scale = 1000 / n;
    const semicolons = (norm.match(/;/g) || []).length * scale;
    const colons = (norm.match(/:/g) || []).length * scale;
    const commas = (norm.match(/,/g) || []).length * scale;
    const dashes = (norm.match(/\u2014|\s-\s/g) || []).length * scale;

    // Flesch-Kincaid Grade
    const fkGrade = 0.39 * meanSentLen + 11.8 * (totalSyllables / n) - 15.59;

    // Key Mosteller-Wallace & Literary markers per 1k
    const rate = (w) => ((counts[w] || 0) * 1000) / n;
    const markers = {
      upon: rate("upon"),
      on: rate("on"),
      while: rate("while"),
      whilst: rate("whilst"),
      enough: rate("enough"),
      consequently: rate("consequently"),
      commonly: rate("commonly"),
      vigor: rate("vigor"),
      there: rate("there"),
      by: rate("by"),
      to: rate("to"),
      from: rate("from")
    };

    // Compute live Federalist Hamilton vs. Madison Log-Likelihood / Delta Score
    // Reference means (per 1k):
    // Hamilton: upon=3.27, on=3.37, while=0.28, whilst=0.005, enough=0.30, consequently=0.02, semicolon=7.10
    // Madison : upon=0.16, on=7.65, while=0.00, whilst=0.32,  enough=0.00, consequently=0.34, semicolon=9.74
    const hFeatures = [markers.upon, markers.on, markers.while * 8, markers.whilst * 8, markers.enough * 8, markers.consequently * 8, semicolons * 0.4];
    const hRef = [3.27, 3.37, 0.28 * 8, 0.005 * 8, 0.30 * 8, 0.02 * 8, 7.10 * 0.4];
    const mRef = [0.16, 7.65, 0.00, 0.32 * 8, 0.00, 0.34 * 8, 9.74 * 0.4];
    const stds = [1.5, 2.2, 1.0, 1.0, 1.0, 1.0, 2.0];

    let distH = 0;
    let distM = 0;
    for (let i = 0; i < hFeatures.length; i++) {
      distH += Math.abs(hFeatures[i] - hRef[i]) / stds[i];
      distM += Math.abs(hFeatures[i] - mRef[i]) / stds[i];
    }
    distH /= hFeatures.length;
    distM /= hFeatures.length;

    return {
      n,
      v,
      meanWordLen: totalChars / n,
      ttr: v / n,
      guiraudR: v / Math.sqrt(n),
      mattr,
      hapaxRatio: v1 / v,
      yulesK,
      honoresR,
      nSents,
      meanSentLen,
      semicolons,
      colons,
      commas,
      dashes,
      fkGrade,
      markers,
      curve,
      distH,
      distM
    };
  }

  function runLiveSandboxAnalysis() {
    const editor = document.getElementById("sandbox-textarea");
    if (!editor) return;
    const stats = computeClientStylometrics(editor.value);
    if (!stats) return;

    // Update metric tiles
    setText("live-tokens", stats.n.toLocaleString());
    setText("live-mattr", stats.mattr.toFixed(3));
    setText("live-yules-k", stats.yulesK.toFixed(1));
    setText("live-honores-r", stats.honoresR.toFixed(0));
    setText("live-sent-len", stats.meanSentLen.toFixed(1) + "w");
    setText("live-semicolons", stats.semicolons.toFixed(2));
    setText("live-upon-on", `${stats.markers.upon.toFixed(2)} / ${stats.markers.on.toFixed(2)}`);
    setText("live-fk-grade", stats.fkGrade.toFixed(1));

    // Length forensic banner
    const banner = document.getElementById("sandbox-length-banner");
    if (banner) {
      if (stats.n < 350) {
        banner.className = "length-banner warn";
        banner.innerHTML = `<strong>Forensic Sample-Size Warning (${stats.n} words):</strong> This passage is under 350 words. High-frequency function words suffer from severe Poisson sampling noise at this length; Eder (2015) recommends ≥2,500 words for high-confidence Delta attribution.`;
      } else {
        banner.className = "length-banner ok";
        banner.innerHTML = `<strong>Sample Size (${stats.n.toLocaleString()} words):</strong> Sufficient token count to observe stable function-word ratios, punctuation cadence (<code>;</code> = ${stats.semicolons.toFixed(1)}/1k), and Mendenhall word-length distribution.`;
      }
    }

    // Live Federalist Marker Attribution Bar
    const closerAuthor = stats.distM <= stats.distH ? "James Madison" : "Alexander Hamilton";
    setText(
      "live-attribution-summary",
      `Closest Publius Profile: ${closerAuthor} (Marker Δ_Madison = ${stats.distM.toFixed(3)} vs. Δ_Hamilton = ${stats.distH.toFixed(3)}) • while/whilst: ${stats.markers.while.toFixed(2)} / ${stats.markers.whilst.toFixed(2)} per 1k`
    );

    drawMendenhallCanvas(stats.curve);
  }

  function drawMendenhallCanvas(liveCurve) {
    const canvas = document.getElementById("mendenhall-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 520;
    const height = 220;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const textColor = isDark ? "#dcd6c8" : "#2b303a";
    const gridColor = isDark ? "rgba(255,255,255,0.09)" : "rgba(0,0,0,0.08)";
    const liveColor = isDark ? "#e67347" : "#b43e22";
    const refColor = isDark ? "#5da9e9" : "#1d568f";

    ctx.clearRect(0, 0, width, height);

    const padL = 42, padR = 18, padT = 18, padB = 32;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const maxY = 0.26;

    // Gridlines
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.fillStyle = textColor;
    ctx.font = "11px JetBrains Mono, monospace";
    for (let yVal = 0; yVal <= 0.25; yVal += 0.05) {
      const y = padT + plotH - (yVal / maxY) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
      ctx.fillText((yVal * 100).toFixed(0) + "%", 6, y + 3);
    }

    // X-axis labels (word lengths 1..15)
    for (let i = 0; i < 15; i++) {
      const x = padL + (i / 14) * plotW;
      ctx.fillText(String(i + 1), x - 4, height - 10);
    }

    const refCurve = DATA.referenceCurves[activeReferenceAuthor] || DATA.referenceCurves["Madison (Federalist)"];

    // Draw reference author curve (dashed)
    ctx.strokeStyle = refColor;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let i = 0; i < 15; i++) {
      const x = padL + (i / 14) * plotW;
      const y = padT + plotH - (Math.min(maxY, refCurve[i]) / maxY) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw live input curve (solid with points)
    ctx.strokeStyle = liveColor;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < 15; i++) {
      const x = padL + (i / 14) * plotW;
      const y = padT + plotH - (Math.min(maxY, liveCurve[i]) / maxY) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = liveColor;
    for (let i = 0; i < 15; i++) {
      const x = padL + (i / 14) * plotW;
      const y = padT + plotH - (Math.min(maxY, liveCurve[i]) / maxY) * plotH;
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function initSandbox() {
    const presetContainer = document.getElementById("sandbox-preset-buttons");
    const textarea = document.getElementById("sandbox-textarea");
    const statusLabel = document.getElementById("sandbox-preset-status");
    const refSelect = document.getElementById("reference-curve-select");

    if (presetContainer && textarea) {
      Object.entries(DATA.presets).forEach(([key, item], idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.id = "preset-btn-" + key;
        btn.className = "btn btn-sm";
        btn.textContent = item.label;
        btn.addEventListener("click", () => {
          textarea.value = item.text;
          if (statusLabel) statusLabel.textContent = "Loaded Preset: " + item.trueStatus;
          runLiveSandboxAnalysis();
        });
        presetContainer.appendChild(btn);
        if (idx === 0) {
          textarea.value = item.text;
          if (statusLabel) statusLabel.textContent = "Loaded Preset: " + item.trueStatus;
        }
      });

      textarea.addEventListener("input", () => {
        if (statusLabel) statusLabel.textContent = "Custom / Edited Text Passage (Live Evaluation)";
        runLiveSandboxAnalysis();
      });
    }

    if (refSelect) {
      Object.keys(DATA.referenceCurves).forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = "Overlay Reference: " + name;
        refSelect.appendChild(opt);
      });
      refSelect.addEventListener("change", () => {
        activeReferenceAuthor = refSelect.value;
        runLiveSandboxAnalysis();
      });
    }

    runLiveSandboxAnalysis();
  }

  // -------------------------------------------------------------------------
  // 4. Disputed Federalist Papers & Unmasking Explorer
  // -------------------------------------------------------------------------
  function renderFederalistSelector() {
    const grid = document.getElementById("disputed-papers-grid");
    if (!grid) return;
    grid.innerHTML = "";
    DATA.disputedFederalist.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.id = "fed-paper-btn-" + p.num;
      btn.className = "filter-btn" + (p.num === activeFederalistNum ? " active" : "");
      btn.textContent = `No. ${p.num} (${p.words}w)`;
      btn.addEventListener("click", () => {
        activeFederalistNum = p.num;
        renderFederalistSelector();
        renderFederalistPaperDetail();
      });
      grid.appendChild(btn);
    });
  }

  function renderFederalistPaperDetail() {
    const p = DATA.disputedFederalist.find((x) => x.num === activeFederalistNum) || DATA.disputedFederalist[2];
    setText("fed-detail-heading", `Federalist No. ${p.num}: ${p.title}`);
    setText("fed-detail-consensus", `Consensus: ${p.consensus} (${p.vote} methods)`);
    setText("fed-detail-bct", `48-Model BCT Stability: ${p.bct}`);
    setText("fed-detail-burrows", `Madison: ${p.bM.toFixed(3)} vs. Hamilton: ${p.bH.toFixed(3)}`);
    setText("fed-detail-cosine", `Madison: ${p.cM.toFixed(3)} vs. Hamilton: ${p.cH.toFixed(3)}`);
    setText("fed-detail-mw", `${(p.mwM * 100).toFixed(1)}% Madison Posterior`);
    setText("fed-detail-gi", `Madison: ${p.giM.toFixed(2)} | Hamilton: ${(1 - p.giM).toFixed(2)}`);
    setText(
      "fed-detail-markers",
      `Observed Marker Rates (per 1,000 words): "upon" = ${p.upon.toFixed(2)} (H avg: 3.27, M avg: 0.16) • "on" = ${p.on.toFixed(2)} (M avg: 7.65) • "whilst" = ${p.whilst.toFixed(2)} • "while" = ${p.while.toFixed(2)}`
    );
  }

  function drawUnmaskingCanvas() {
    const canvas = document.getElementById("unmasking-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 520;
    const height = 225;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const textColor = isDark ? "#dcd6c8" : "#2b303a";
    const gridColor = isDark ? "rgba(255,255,255,0.09)" : "rgba(0,0,0,0.08)";
    const diffColor = isDark ? "#e67347" : "#b43e22"; // Disputed vs Hamilton (remains high)
    const sameColor = isDark ? "#52b788" : "#1f7a54"; // Disputed vs Madison (degrades fast)
    const ctrlColor = isDark ? "#5da9e9" : "#1d568f"; // Same-author control

    ctx.clearRect(0, 0, width, height);
    const padL = 44, padR = 20, padT = 18, padB = 32;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Y-axis (0.35 to 1.05)
    ctx.strokeStyle = gridColor;
    ctx.fillStyle = textColor;
    ctx.font = "11px JetBrains Mono, monospace";
    for (let acc = 0.4; acc <= 1.0; acc += 0.2) {
      const y = padT + plotH - ((acc - 0.35) / 0.7) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
      ctx.fillText((acc * 100).toFixed(0) + "%", 6, y + 3);
    }

    const rounds = DATA.unmaskingCurves.rounds;
    for (let i = 0; i < rounds.length; i++) {
      const x = padL + (i / (rounds.length - 1)) * plotW;
      ctx.fillText("R" + rounds[i], x - 7, height - 10);
    }

    function plotSeries(arr, color, dashed) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.4;
      ctx.setLineDash(dashed ? [5, 4] : []);
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const x = padL + (i / (arr.length - 1)) * plotW;
        const y = padT + plotH - ((arr[i] - 0.35) / 0.7) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    plotSeries(DATA.unmaskingCurves.disputed_vs_hamilton, diffColor, false);
    plotSeries(DATA.unmaskingCurves.hamilton_split_same, ctrlColor, true);
    plotSeries(DATA.unmaskingCurves.disputed_vs_madison, sameColor, false);
  }

  // -------------------------------------------------------------------------
  // 5. Interactive Sample-Size Floor Simulator (Use & Limits Module)
  // -------------------------------------------------------------------------
  function initSampleSizeSimulator() {
    const slider = document.getElementById("sample-size-slider");
    if (!slider) return;

    function updateSim() {
      const words = Number(slider.value);
      setText("sim-word-count-display", words.toLocaleString() + " words");

      // Empirical Eder (2015) / Evert (2017) accuracy scaling curves as a function of sample length
      const cosineAcc = Math.min(99.6, 52 + 47.6 * (1 - Math.exp(-words / 1150)));
      const burrowsAcc = Math.min(98.2, 48 + 50.2 * (1 - Math.exp(-words / 1350)));
      const mendAcc = Math.min(76.0, 38 + 38.0 * (1 - Math.exp(-words / 2200)));
      const noiseMargin = Math.max(1.2, 28 * Math.exp(-words / 1200));

      setText("sim-cosine-acc", `${cosineAcc.toFixed(1)}% (±${noiseMargin.toFixed(1)}%)`);
      setText("sim-burrows-acc", `${burrowsAcc.toFixed(1)}% (±${(noiseMargin * 1.15).toFixed(1)}%)`);
      setText("sim-mend-acc", `${mendAcc.toFixed(1)}%`);

      let verdict = "";
      if (words < 500) {
        verdict = "CRITICAL WARNING (<500w): Unreliable for forensic attribution. Over 70% of MFW function words have 0 or 1 occurrences; random topic noise dominates.";
      } else if (words < 1500) {
        verdict = "CAUTION (500–1,500w): Moderate accuracy. Short essays or letters require Mosteller–Wallace marker words, Eder's Delta, and General Imposters verification.";
      } else if (words < 3000) {
        verdict = "RELIABLE THRESHOLD (1,500–3,000w): Standard length of a Federalist essay (~2,000w). Cosine Delta and Ensemble Consensus exceed 94% accuracy.";
      } else {
        verdict = "HIGH-CONFIDENCE REGIME (≥3,000w): Book-chapter length (Eder's recommended zone). Cosine Delta, Burrows's Delta, and General Imposters approach 98–100% accuracy.";
      }
      setText("sim-verdict-text", verdict);
    }

    slider.addEventListener("input", updateSim);
    updateSim();
  }

  // -------------------------------------------------------------------------
  // 6. Render Limits Cards
  // -------------------------------------------------------------------------
  function renderLimitsGrid() {
    const grid = document.getElementById("limits-cards-grid");
    if (!grid) return;
    grid.innerHTML = "";
    DATA.limitsGuide.forEach((item) => {
      const card = document.createElement("article");
      card.className = "limit-card";
      card.innerHTML = `
        <span class="limit-card-num">${escapeHtml(item.num)}</span>
        <h3>${escapeHtml(item.title)}</h3>
        <p style="font-size: 0.9rem;">${escapeHtml(item.summary)}</p>
        <div class="limit-mitigation">
          <strong style="color: var(--accent-success);">Best-Practice Mitigation:</strong> ${escapeHtml(item.mitigation)}
        </div>
      `;
      grid.appendChild(card);
    });
  }

  // Helpers
  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Boot
  window.addEventListener("DOMContentLoaded", () => {
    initTheme();
    renderCategoryFilters();
    renderMethodsList();
    renderMethodDossier();
    initSandbox();
    renderFederalistSelector();
    renderFederalistPaperDetail();
    drawUnmaskingCanvas();
    initSampleSizeSimulator();
    renderLimitsGrid();

    window.addEventListener("resize", () => {
      runLiveSandboxAnalysis();
      drawUnmaskingCanvas();
    });
  });
})();
