# IEEE paper: Calibrated Concept Bottleneck Models on a Frozen Spiking Vision Backbone

Files

- `main.tex`: the paper (IEEEtran conference format, two columns).
- `refs.bib`: references (39 entries; checked against the original papers).
- `figs/`: the three result figures (vector PDF). Figure 1 is drawn in `main.tex` with TikZ.
- `make_figures.py`: regenerates `figs/`.
  - Every number in it is copied from a report in this repository; the source is named in each block.
  - The reliability diagram is recomputed from the saved test outputs.
  - Run: `python make_figures.py "<path to SpikingResformer-cbm>"`.
- `main.pdf`: the compiled paper.

Compile

- **Overleaf:** New Project > Upload Project > upload the zip. Then click Recompile; pdfLaTeX is the default compiler.
- **Locally:** `latexmk -pdf main.tex`.

Fill in before submitting

1. Department, institution, city and e-mail for each author.
   - All four authors currently carry `[Department]`, `[Institution]` and `[email]` placeholders.
2. The guide's name in the Acknowledgment.
   - Add the guide as an author if that is your department's practice.
3. The AI-use disclosure in the Acknowledgment.
   - IEEE requires authors to disclose AI-generated content; edit the sentence to match your actual use.
4. The target venue's page limit and template rules.
   - The draft is 8 pages including references.

Where the numbers come from

All numbers come from the repository reports listed in `REVIEW_RESULTS.md` (section 7). The reports for these parts of the paper are:

- The per-image stability test (Section V-G, Table V): `stability_test.py`, with results in `stability/report.md` and `stability/report.json`.
- The patched-neuron check (Section III-A): `hook_check.py`, with results in `hook_check/report.md` and `hook_check/report.json`.
- The NEC comparison with VLG-CBM: `nec_sparse/nec_sparse_report.md`.
