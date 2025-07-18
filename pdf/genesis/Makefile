# Makefile for Genesis Unified Theory PDF

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex
MAKEINDEX = makeindex

# Main document
MAIN = main

# All tex files
TEX_FILES = $(wildcard *.tex)

# Default target
all: $(MAIN).pdf

# Build PDF
$(MAIN).pdf: $(TEX_FILES)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN) || true
	$(MAKEINDEX) $(MAIN) || true
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.idx *.ind *.ilg *.lof *.lot

# Clean all including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# View PDF
view: $(MAIN).pdf
	open $(MAIN).pdf || xdg-open $(MAIN).pdf

# Quick build (single pass)
quick:
	$(LATEX) $(MAIN)

.PHONY: all clean cleanall view quick