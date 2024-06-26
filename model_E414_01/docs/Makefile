# This Makefile is based somewhat on the Makefile present here:
# https://github.com/tompollard/phd_thesis_markdown/blob/main/Makefile

PANDOC=pandoc

BASEDIR=$(CURDIR)
INPUTDIR=$(BASEDIR)/source
OUTPUTDIR=$(BASEDIR)/output
BIBFILE=$(INPUTDIR)/references.bib
STYLEDIR=$(BASEDIR)/style

PREAMBLE=$(BASEDIR)/../../../Personal-Templates/pandoc/styles/preamble.latex
GLOSSARY=$(BASEDIR)/../../../Personal-Templates/pandoc/styles/glossary.latex
TEMPLATE=$(BASEDIR)/../../../Personal-Templates/pandoc/styles/template.latex
CSL=$(BASEDIR)/../../../Personal-Templates/pandoc/styles/csls/ieee-with-url.csl
FILTERS=$(BASEDIR)/../../../Personal-Templates/pandoc/filters

pdf:
	pandoc  \
		--output "$(OUTPUTDIR)/out.pdf" \
		--template="$(TEMPLATE)" \
		--include-in-header="$(GLOSSARY)" \
		--include-in-header="$(PREAMBLE)" \
		--pdf-engine=xelatex \
		--pdf-engine-opt=-shell-escape \
		"$(INPUTDIR)"/*.md \
		"$(INPUTDIR)/metadata.yml" \
		--lua-filter="$(FILTERS)/bquote.lua" \
		--lua-filter="$(FILTERS)/fig.lua" \
		--bibliography="$(BIBFILE)" \
		--citeproc \
		--csl="$(CSL)"

all: pdf

.PHONY: help install pdf docx html tex
