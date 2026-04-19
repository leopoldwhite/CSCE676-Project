# CSCE 676 Course Project — Graph-Augmented Reasoning with GRBench

**Course**: CSCE 676, Data Mining and Analysis — Spring 2026
**Author**: Yuyang Bai (`ybai@tamu.edu`)

This repository contains the full semester project: a three-research-question study on the GRBench multi-hop reasoning benchmark, combining **text mining + clustering (RQ1)**, **graph mining (RQ2)**, and a beyond-course method — **graph-aware reinforcement learning with a biased-mixture curriculum (RQ3)**.

## TL;DR — headline findings

- **RQ1 (Text mining + clustering).** Sentence-BERT (`all-MiniLM-L6-v2`) embeddings lift macro-F1 on difficulty prediction from **0.33 (domain-label-only baseline) to 0.83 (logistic regression)**, a >2× improvement significant at **p ≤ 7.5e-05** under a corrected resampled paired t-test. But unsupervised K-Means recovers *domain* structure (ARI 0.265) — not *difficulty* (ARI 0.044). **Conclusion: embedding geometry encodes topic, not reasoning complexity.**
- **RQ2 (Graph mining).** Keyword co-occurrence graphs show **significant negative correlation between centrality and question difficulty** (Spearman r = −0.160 for PageRank, r = −0.131 for betweenness; both p < 1e-7 on n = 1,740). A TF-IDF baseline beats dense embeddings on difficulty prediction (macro-F1 **0.926 vs 0.832**). **Conclusion: harder questions invoke the long tail of the vocabulary — dense embeddings compress away part of the signal.**
- **RQ3 (Beyond course — Graph-aware curriculum RL).** A biased-mixture curriculum on Qwen2.5-3B-Instruct achieves **Rouge-L 40.62 / GPT4Score 42.25** averaged over 4 held-out OOD domains, beating vanilla PPO RL by **+2.13 Rouge-L / +3.20 GPT4Score** and matching a prompted Qwen3-14B (39.85 / 42.01) at one-fifth the parameter count. Largest gains land exactly where RQ1/RQ2 predicted difficulty lives (Legal-Hard: +11.06 Rouge-L). **Conclusion: curriculum beats scale on structured-knowledge QA.**

## For graders — where to start

| Artefact | What it shows |
|---|---|
| [`notebooks/final_deliverable.ipynb`](notebooks/final_deliverable.ipynb) | **Start here.** Curated end-to-end story: motivation → RQ1 → RQ2 → RQ3 → cross-RQ synthesis → limitations. 38 cells (21 markdown / 17 code), all executed, full outputs inline. |
| [`handoff/SUMMARY.md`](handoff/SUMMARY.md) | Frozen experimental artefact index + per-RQ detail (numbers, tables, figures, open risks). The final notebook reads from this. |
| [`handoff/reproducibility.md`](handoff/reproducibility.md) | Commits, seeds, hardware, wall-clock times. |
| [`notebooks/checkpoint1_dataset_selection.ipynb`](notebooks/checkpoint1_dataset_selection.ipynb) · [`notebooks/checkpoint2_rq_formation.ipynb`](notebooks/checkpoint2_rq_formation.ipynb) | Checkpoint 1 and Checkpoint 2 notebooks (history, not required for grading the final). |

The final notebook's top cell contains the **collaboration declaration** (human collaborators, web sources, AI tools, full citation list) per the course rubric.

## Repository structure

```
CSCE676-Project/
├── notebooks/
│   ├── final_deliverable.ipynb              ← the final submission
│   ├── checkpoint1_dataset_selection.ipynb  ← Checkpoint 1
│   └── checkpoint2_rq_formation.ipynb       ← Checkpoint 2
├── handoff/                                 ← frozen experimental artefacts read by the final notebook
│   ├── SUMMARY.md                           ← one-page overview of all three RQs
│   ├── reproducibility.md                   ← commits / seeds / hardware / wall-clock
│   ├── environment.txt                      ← pip freeze (190 packages)
│   ├── rq1_text_clustering/                 ← notebook, metrics.json, figures, CSV tables
│   ├── rq2_graph_mining/                    ← notebook, metrics.json, figures, CSV tables
│   └── rq3_graph_aware_rl/                  ← notebook, metrics.json, figures, CSV tables, qualitative trajectories
├── scripts/
│   └── download_datasets.py                 ← one-command dataset downloader
├── reports/
│   └── figures/                             ← EDA and analysis figures
└── data/raw/                                ← downloaded raw datasets (git-ignored)
```

## Reproducibility quickstart

The final notebook runs entirely from the frozen `handoff/` artefacts — it does **not** re-execute training or inference. A grader can reproduce the final notebook end-to-end with only lightweight Python libraries:

```bash
git clone https://github.com/leopoldwhite/CSCE676-Project.git
cd CSCE676-Project

# Create a venv (uv recommended, stdlib venv also works)
uv venv .venv
uv pip install --python .venv/bin/python pandas numpy matplotlib seaborn jupyter nbformat nbclient

# Re-execute the final deliverable notebook in place (produces identical outputs)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/final_deliverable.ipynb
```

To additionally re-run Checkpoint 1's EDA (optional — requires downloading GRBench):

```bash
python3 scripts/download_datasets.py --data-root data/raw --log-path reports/dataset_download_log.json
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/checkpoint1_dataset_selection.ipynb
```

Exact commits, seeds, and hardware for the RQ3 training runs (not re-executed in this repo) are listed in [`handoff/reproducibility.md`](handoff/reproducibility.md).

## Research questions

1. **RQ1 (Course — Text Mining + Clustering):** Can text embeddings and unsupervised clustering reveal latent question types that predict difficulty level better than domain labels alone?
2. **RQ2 (Course — Graph Mining):** What structural properties of keyword co-occurrence graphs differentiate domains, and do graph centrality measures correlate with question difficulty?
3. **RQ3 (External — Graph-Aware RL):** Can graph-aware reinforcement learning with a difficulty-escalating curriculum improve multi-hop reasoning trajectories on GRBench compared to prompting-only and uniform-sampling RL baselines?

## Dataset

**GRBench** ([HuggingFace dataset card](https://huggingface.co/datasets/PeterJinGo/GRBench)) — 1,740 question–answer pairs across 10 textual-graph domains with author-assigned difficulty labels (easy / medium / hard). Selected over ogbn-arxiv and SNAP com-Amazon because it is the only candidate that simultaneously supports the text-mining / graph-mining / graph-reasoning triad required by the project and allows a meaningful beyond-course RL methodology. Full selection rationale and comparative analysis is in `notebooks/checkpoint1_dataset_selection.ipynb`.

## Checkpoint history

- **Checkpoint 1 (Dataset Selection + EDA)** — candidate dataset triage (GRBench / ogbn-arxiv / SNAP com-Amazon), final selection, exploratory data analysis, initial research questions.
- **Checkpoint 2 (Research Question Formation)** — three locked RQs with method mapping, additional EDA (UMAP, keyword co-occurrence, reasoning complexity), initial method runs (K-Means/DBSCAN, Louvain, centrality, difficulty baseline), feasibility analysis.
- **Final Deliverable** — this README and `notebooks/final_deliverable.ipynb`.

## Data and licensing

- **GRBench**: Apache-2.0 (per HF dataset card).
- **ogbn-arxiv**: subject to OGB benchmark terms and required citations (considered but not selected).
- **SNAP com-Amazon**: subject to SNAP data usage and citation norms (considered but not selected).

All figures and tables in this repository were generated by the author from the above data. Full citation list is included at the top of `notebooks/final_deliverable.ipynb`.
