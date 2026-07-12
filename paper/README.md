# IEEE Paper — Decoding Currency Dynamics

`Decoding_Currency_Dynamics_IEEE.tex` is a complete IEEE conference paper written in the
official **IEEEtran** LaTeX class. It compiles to a standard two-column IEEE PDF.

## Easiest: Overleaf (recommended, no install)
1. Go to <https://www.overleaf.com> → **New Project → Upload Project** (or **Blank Project**).
2. Upload `Decoding_Currency_Dynamics_IEEE.tex`.
3. Overleaf ships `IEEEtran.cls` — just press **Recompile**. Done.

Alternatively start from Overleaf's **"IEEE Conference Template"** and paste in the body.

## Local compile
**Tectonic** (single self-contained binary, auto-downloads IEEEtran):
```bash
brew install tectonic        # macOS
tectonic Decoding_Currency_Dynamics_IEEE.tex
```
**Full TeX Live / MacTeX:**
```bash
pdflatex Decoding_Currency_Dynamics_IEEE
pdflatex Decoding_Currency_Dynamics_IEEE   # run twice for references
```

## Before submission — please complete
- **References [6]–[9]** are the survey papers cited from the dissertation; fill in the
  exact authors, titles, venue and DOI from your Literature Survey.
- Replace the placeholder architecture box (Fig. 1) with the exported architecture figure
  (e.g. a PNG from the dashboard/report) via `\includegraphics`.
- Confirm the author block, affiliation and any co-authors/supervisor.
- Numbers in the tables are the committed results (Hybrid DirAcc 0.534, GARCH 0.575,
  ARIMA 0.485; ablation 0.5006 / 0.5209 / 0.5345). Update if you re-run.
- Match the target venue's template (conference vs journal `\documentclass` option) and
  page limit.
