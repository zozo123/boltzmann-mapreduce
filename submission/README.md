# Flat submission package

This directory contains the complete submission package for **Evidence-Aware
MapReduce for Forkable Compute: Separating Execution Fan-Out from Independent
Evidence**. Every source and build artifact sits at the top level so the package
can be uploaded as-is.

Files:

- `main.tex` — document preamble and entry point.
- `body.tex` — manuscript text, equations, and tables.
- `fig_contract.tex` — code-native architecture figure.
- `refs.bib` — cited bibliography.
- `evidence-aware-reduction.pdf` — reproducible built manuscript.

Build from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The repository-level `scripts/build_paper.py` performs the same build in a
temporary directory, enforces release warnings, fixes reproducible metadata, and
publishes byte-identical PDF copies.
