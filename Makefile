# Minimal makefile for Sphinx documentation
SPHINXOPTS    =
SPHINXBUILD   = python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

# Generate matplotlib figures before building
GENERATE_FIGURES = python3 source/figures/generate_all.py

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile figures

figures:
	@echo "Generating matplotlib figures..."
	@$(GENERATE_FIGURES)

%: Makefile
	@$(MAKE) figures
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
