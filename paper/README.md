# Paper draft

LaTeX source for the CVPR EarthVision 2027 workshop submission.

## Build

```bash
# 1. Drop the NeurIPS 2024 style file in this directory:
curl -O https://media.neurips.cc/Conferences/NeurIPS2024/Styles/neurips_2024.sty

# 2. Compile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The skeleton mirrors the `arun-thesis` template per global LaTeX preferences.
Replace placeholder citations and tables with final numbers post-training.
