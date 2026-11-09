# R.E.A.P — Major Project End-Semester Report

This folder contains the complete, self-contained LaTeX source of the
Major Project End-Semester report.

## Contents

| File | Purpose |
|------|---------|
| `Major_Project_Endsem_Report.tex` | Main LaTeX source (authoritative) |
| `symbol.jpg` | NITK emblem used on the cover page |
| `Major_Project_Endsem_Report.pdf` | Compiled report (42 pages) |
| `build_calibrated_report.py` | Optional Python generator that reproduces the `.tex` |
| `build.bat` | Two-pass `pdflatex` build script (Windows) |

No external `.bib` file is needed — the bibliography is embedded with
`thebibliography` inside the `.tex`.

## Compile

From this folder:

```bat
pdflatex -interaction=nonstopmode Major_Project_Endsem_Report.tex
pdflatex -interaction=nonstopmode Major_Project_Endsem_Report.tex
```

or simply run `build.bat`. Run `pdflatex` twice so that the table of contents,
list of figures, list of tables, and cross-references resolve correctly.

The document is standard `pdflatex` (no XeLaTeX/LuaLaTeX, no shell escape).

## TeX packages

The report uses the NITK report template packages plus the following. If you
use a minimal TeX installation (e.g. TinyTeX), install them with:

```bat
tlmgr install pdfpages rotfloat subfig algorithm2e enumitem titlesec ^
  sectsty comment soul cancel chngcntr pgfplots pbox appendix setspace ^
  tocbibind framed booktabs colortbl listings ragged2e babel-english ^
  ifoddpage relsize
```

Packages such as `lscape`, `afterpage`, `tabularx`, `xspace`, `verbatim`,
`float`, `geometry`, `graphicx`, `caption`, `hyperref`, `amsmath`, `amssymb`,
`babel`, `tikz`, `multirow`, `array`, `url`, `xcolor`, `setspace`, and
`titlesec` are part of the standard TeX Live collections.

## Optional regeneration

`build_calibrated_report.py` rebuilds the `.tex` from its embedded text blocks:

```bat
python build_calibrated_report.py
```

It requires Python 3 and PyMuPDF (`pip install pymupdf`). The generated `.tex`
is content-identical to the checked-in `Major_Project_Endsem_Report.tex`.

## Changes applied for the end-semester submission

1. Section 1.2 starts on a new page; its lead-in now reads
   “The details are as follows:” and all four exercise points appear together.
2. Widow/orphan penalties were enabled (`\widowpenalty`, `\clubpenalty`,
   `\displaywidowpenalty`) and specific page breaks were adjusted so that no
   single line of a paragraph is stranded at the top or bottom of a page.
3. The Problem Statement in Section 2.8 is now a single statement.
4. Citations were added throughout the introduction, literature review,
   methodology, and results; all 30 bibliography entries are cited.
5. Figure and table captions were shortened.
6. Research objectives are labelled `1.`–`5.` instead of `RO 1`–`RO 5`.
7. Figures, tables, and algorithms are numbered chapter-wise
   (e.g. `3.1`, `3.2`) instead of subsection-wise (`3.1.1`).
8. Algorithms are numbered by chapter (`Algorithm 3.1`, `Algorithm 3.2`) and
   are cited in the text.
9. Landmark counts `(33)` and `(17)` were removed; the text now refers simply to
   MediaPipe BlazePose and COCO keypoints.
10. Section 5.1 is now **Conclusions**, followed by the summary, technical
    contributions, limitations, and future work.
11. Semester 8 work is presented as **Future Work** (Section 5.5).
