# CSCE 676 Course Project — Graph-Augmented Reasoning with GRBench

**Course**: CSCE 676, Data Mining and Analysis — Spring 2026
**Author**: Yuyang Bai (`ybai@tamu.edu`)

> 👉 **Start here:** [`notebooks/main_notebook.ipynb`](notebooks/main_notebook.ipynb) — the curated end-to-end story, fully executed with all outputs inline.
>
> 🎥 **2-minute project video:** https://www.youtube.com/watch?v=8UMWeh24A3E

This repository contains the full semester project: a three-research-question study on the GRBench multi-hop reasoning benchmark, combining **text mining + clustering (RQ1)**, **graph mining (RQ2)**, and a beyond-course method — **graph-aware reinforcement learning with a biased-mixture curriculum (RQ3)**.

## TL;DR — headline findings

- **RQ1 (Text mining + clustering).** Sentence-BERT (`all-MiniLM-L6-v2`) embeddings lift macro-F1 on difficulty prediction from **0.33 (domain-label-only baseline) to 0.83 (logistic regression)**, a >2× improvement significant at **p ≤ 7.5e-05** under a corrected resampled paired t-test. But unsupervised K-Means recovers *domain* structure (ARI 0.265) — not *difficulty* (ARI 0.044). **Conclusion: embedding geometry encodes topic, not reasoning complexity.**
- **RQ2 (Graph mining).** Keyword co-occurrence graphs show **significant negative correlation between centrality and question difficulty** (Spearman r = −0.160 for PageRank, r = −0.131 for betweenness; both p < 1e-7 on n = 1,740). A TF-IDF baseline beats dense embeddings on difficulty prediction (macro-F1 **0.926 vs 0.832**). **Conclusion: harder questions invoke the long tail of the vocabulary — dense embeddings compress away part of the signal.**
- **RQ3 (Beyond course — Graph-aware curriculum RL).** A biased-mixture curriculum on Qwen2.5-3B-Instruct achieves **Rouge-L 40.62 / GPT4Score 42.25** averaged over 4 held-out OOD domains, beating vanilla PPO RL by **+2.13 Rouge-L / +3.20 GPT4Score** and matching a prompted Qwen3-14B (39.85 / 42.01) at one-fifth the parameter count. Largest gains land exactly where RQ1/RQ2 predicted difficulty lives (Legal-Hard: +11.06 Rouge-L). **Conclusion: curriculum beats scale on structured-knowledge QA.**

## Research questions

1. **RQ1 (Course — Text Mining + Clustering):** Can text embeddings and unsupervised clustering reveal latent question types that predict difficulty level better than domain labels alone?
2. **RQ2 (Course — Graph Mining):** What structural properties of keyword co-occurrence graphs differentiate domains, and do graph centrality measures correlate with question difficulty?
3. **RQ3 (External — Graph-Aware RL):** Can graph-aware reinforcement learning with a difficulty-escalating curriculum improve multi-hop reasoning trajectories on GRBench compared to prompting-only and uniform-sampling RL baselines?

## Data

**GRBench** ([HuggingFace dataset card](https://huggingface.co/datasets/PeterJinGo/GRBench)) — 1,740 question–answer pairs across 10 textual-graph domains (academic citations, Amazon products, literature, biomedical, fiction, etc.) with author-assigned difficulty labels (easy / medium / hard). Selected over `ogbn-arxiv` and SNAP `com-Amazon` because it is the only candidate that simultaneously supports the text-mining / graph-mining / graph-reasoning triad required by the project and allows a meaningful beyond-course RL methodology. Full selection rationale and comparative analysis is in `checkpoints/checkpoint_1.ipynb`.

**Where to get it.** Run `scripts/download_datasets.py` (see Reproducibility below). The raw dataset is *not* committed to the repo (gitignored under `data/raw/`).

**Preprocessing.** The final notebook does not preprocess raw GRBench at runtime — instead, it reads frozen experimental artefacts from `handoff/` (CSV tables, JSON metrics, PNG figures). Preprocessing for each RQ — Sentence-BERT encoding (RQ1), keyword co-occurrence graph construction (RQ2), training/inference logs (RQ3) — was performed offline and the outputs are committed under `handoff/rq{1,2,3}_*/`. See [`handoff/reproducibility.md`](handoff/reproducibility.md) for upstream commits, seeds, and hardware.

## How to reproduce

The final notebook reads entirely from the frozen `handoff/` artefacts — it does **not** re-execute training or inference, so it can be rerun anywhere with lightweight Python libraries (no GPU required).

**Option A — Google Colab (zero setup).** Open `notebooks/main_notebook.ipynb` in Colab, run the first cell to clone the repo and install dependencies:

```python
!git clone https://github.com/leopoldwhite/CSCE676-Project.git
%cd CSCE676-Project
!pip install -r requirements.txt
```

Then **Runtime → Run all**.

**Option B — Local with `uv` (the environment used by the author).**

```bash
git clone https://github.com/leopoldwhite/CSCE676-Project.git
cd CSCE676-Project

uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Re-execute the main notebook in place (produces identical outputs)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/main_notebook.ipynb
```

**Run order**: only `notebooks/main_notebook.ipynb` is required for grading. The two checkpoint notebooks (`checkpoints/checkpoint_1.ipynb`, `checkpoints/checkpoint_2.ipynb`) are included for the progression-of-work story; they were executed earlier in the semester and do not need to be re-run.

To additionally re-run Checkpoint 1's EDA (optional — requires downloading raw GRBench):

```bash
python3 scripts/download_datasets.py --data-root data/raw --log-path reports/dataset_download_log.json
.venv/bin/jupyter nbconvert --to notebook --execute --inplace checkpoints/checkpoint_1.ipynb
```

## Key dependencies and versions

- **Python**: 3.10+ (developed on Python 3.10.12 / 3.11; works on Colab's default Python 3.11)
- `pandas` ≥ 2.0 (tested 2.3.3)
- `numpy` ≥ 1.24 (tested 2.2.6)
- `matplotlib` ≥ 3.7 (tested 3.10.8)
- `jupyter` / `nbconvert` / `nbclient` (any modern version) — only needed if you want to re-execute the notebook

The full pin list is in [`requirements.txt`](requirements.txt) at the repo root.

## Repository structure

```
CSCE676-Project/
├── README.md                                ← you are here
├── requirements.txt                         ← pinned runtime dependencies
├── notebooks/
│   └── main_notebook.ipynb                  ← 👉 the final submission (curated story)
├── checkpoints/
│   ├── checkpoint_1.ipynb                   ← Checkpoint 1: dataset selection + EDA
│   └── checkpoint_2.ipynb                   ← Checkpoint 2: research-question formation
├── handoff/                                 ← frozen experimental artefacts read by main_notebook.ipynb
│   ├── SUMMARY.md                           ← one-page overview of all three RQs
│   ├── reproducibility.md                   ← commits / seeds / hardware / wall-clock
│   ├── environment.txt                      ← full pip freeze of the experiment env (190 packages)
│   ├── rq1_text_clustering/                 ← notebook, metrics.json, figures, CSV tables
│   ├── rq2_graph_mining/                    ← notebook, metrics.json, figures, CSV tables
│   └── rq3_graph_aware_rl/                  ← notebook, metrics.json, figures, CSV tables, qualitative trajectories
├── scripts/
│   └── download_datasets.py                 ← one-command dataset downloader
├── reports/
│   └── figures/                             ← EDA and analysis figures (also embedded in notebooks)
└── data/raw/                                ← downloaded raw datasets (git-ignored)
```

The collaboration declaration (human collaborators, web sources, AI tools, full citation list) is the first cell of `notebooks/main_notebook.ipynb`.

## Results summary

The three RQs answer the same underlying question — *what does it take to predict or solve difficulty on GRBench?* — from three angles. RQ1 shows that semantic embeddings beat domain labels for difficulty prediction (macro-F1 0.33 → 0.83) but unsupervised clustering recovers only topic, not difficulty. RQ2 sharpens this structurally: harder questions significantly invoke peripheral vocabulary (Spearman r = −0.160, p < 1e-10), and a TF-IDF baseline unexpectedly beats dense embeddings on the same task (0.926 vs 0.832). RQ3 then shows that when a 3B LLM is trained with a biased-mixture curriculum to interleave reasoning with graph function calls, it learns evidence-grounded multi-hop reasoning that beats vanilla PPO and matches a prompted 14B model on Rouge-L / GPT4Score, with the largest gains exactly where the prior RQs predicted difficulty lives (Legal-Hard: +11.06 Rouge-L). The full headline figure is in `notebooks/main_notebook.ipynb`'s "Cross-RQ synthesis" section.

## Data and licensing

- **GRBench**: Apache-2.0 (per HF dataset card).
- **ogbn-arxiv**: subject to OGB benchmark terms and required citations (considered but not selected).
- **SNAP com-Amazon**: subject to SNAP data usage and citation norms (considered but not selected).

All figures and tables in this repository were generated by the author from the above data. Full citation list is included at the top of `notebooks/main_notebook.ipynb`.
