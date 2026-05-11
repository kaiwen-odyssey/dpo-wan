#!/usr/bin/env bash
# Compile the dpo-wan paper PDF.  Run after `scripts/07_render_paper.py`
# has produced tables/ and figures/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p tables figures

# placeholder tables/figures so LaTeX compiles even before results are in
[[ -f tables/main.tex      ]] || printf '%s\n' '\textit{(pending)}' > tables/main.tex
[[ -f tables/ablation.tex  ]] || printf '%s\n' '\textit{(pending)}' > tables/ablation.tex
[[ -f tables/scale.tex     ]] || printf '%s\n' '\textit{(pending)}' > tables/scale.tex

PY="${REPO_PY:-../.venv/bin/python}"
if [[ ! -f figures/radar.pdf ]]; then
  ${PY} -c "
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.figure(); plt.text(0.5, 0.5, '(no data)', ha='center', va='center')
plt.axis('off'); plt.savefig('figures/radar.pdf'); plt.close()
"
fi
if [[ ! -f figures/loss_curves.pdf ]]; then
  ${PY} -c "
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.figure(); plt.text(0.5, 0.5, '(no data)', ha='center', va='center')
plt.axis('off'); plt.savefig('figures/loss_curves.pdf'); plt.close()
"
fi

pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/tmp/pdflatex.log 2>&1 || \
    (tail -40 /tmp/pdflatex.log; exit 1)
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/tmp/pdflatex.log 2>&1 || \
    (tail -40 /tmp/pdflatex.log; exit 1)

ls -la paper.pdf
