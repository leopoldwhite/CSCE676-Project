# CSCE 676 Course Project — Graph-Augmented Reasoning with GRBench

This repository contains the semester project for CSCE 676 (Data Mining and Analysis, Spring 2026). The project investigates text mining, graph mining, and graph-aware reinforcement learning on the GRBench benchmark.

## Checkpoints

### Checkpoint 1: Dataset Selection and EDA
- (A) Candidate dataset identification (3 datasets, including GRBench)
- (B) Comparative analysis table with required dimensions
- (C) Final dataset selection + trade-offs
- (D) EDA on selected dataset (GRBench)
- (E) Initial insights + research questions
- (F) GitHub portfolio setup artifacts

### Checkpoint 2: Research Question Formation
- Project scope recap with EDA findings
- Three research questions with method mapping (2 course + 1 external)
- Additional EDA: UMAP embeddings, keyword co-occurrence graphs, reasoning complexity
- Initial method runs: K-Means/DBSCAN clustering, Louvain community detection, centrality analysis, difficulty classification baseline
- Motivation and feasibility analysis for each RQ
- Methodological planning with timeline

### Final Deliverable
- `notebooks/final_deliverable.ipynb` — curated end-to-end story: motivation → RQ1 (text mining + clustering) → RQ2 (graph mining) → RQ3 (graph-aware curriculum RL) → headline findings → honest limitations.
- `handoff/` — frozen experimental artefacts the notebook reads from (metrics JSONs, tables, figures, environment manifest, reproducibility notes). The notebook curates these artefacts; it does not re-run the full training pipeline.
- Collaboration declaration (human collaborators, web sources, AI tools, full citations) is included at the top of the notebook per the course rubric.

## Candidate Datasets

1. **GRBench**  
   Source: <https://huggingface.co/datasets/PeterJinGo/GRBench>

2. **ogbn-arxiv**  
   Source: <https://ogb.stanford.edu/docs/nodeprop/>

3. **SNAP com-Amazon**  
   Source: <https://snap.stanford.edu/data/com-Amazon.html>

## Selected Dataset

**GRBench** is selected for this checkpoint because it most directly supports the graph-reasoning project direction while still covering course topics and enabling beyond-course techniques (graph-aware RL for tool-using LLM trajectories).

## Explicit Course Alignment for GRBench

- Text mining (course): questions/answers support lexical and embedding-based analyses.
- Graph reasoning/mining (course): benchmark is built around multi-hop reasoning over textual graphs, with difficulty labels for structured analysis.
- Clustering (course, if covered): question embeddings can be clustered to analyze intent families and difficulty patterns.
- Beyond-course method: graph-aware RL for tool-use trajectory optimization.

## Research Questions

1. **RQ1 (Course — Text Mining + Clustering):** Can text embeddings and unsupervised clustering reveal latent question types that predict difficulty level better than domain labels alone?
2. **RQ2 (Course — Graph Mining):** What structural properties of keyword co-occurrence graphs differentiate domains, and do graph centrality measures correlate with question difficulty?
3. **RQ3 (External — Graph-Aware RL):** Can graph-aware reinforcement learning improve multi-hop reasoning trajectories on GRBench compared to prompting-only baselines?

## Repository Structure

- `notebooks/checkpoint1_dataset_selection.ipynb` — Checkpoint 1 notebook
- `notebooks/checkpoint2_rq_formation.ipynb` — Checkpoint 2 notebook
- `notebooks/final_deliverable.ipynb` — **Final deliverable**: end-to-end curated story across RQ1 / RQ2 / RQ3
- `handoff/` — frozen experimental artefacts consumed by the final deliverable notebook (metrics, tables, figures, reproducibility notes)
- `scripts/download_datasets.py` — one-command dataset downloader
- `reports/figures/` — EDA and analysis figures
- `data/raw/` — downloaded raw datasets (local use, not committed)

## Quickstart

```bash
cd course_project_checkpoint1
uv venv .venv
uv pip install --python .venv/bin/python pandas matplotlib seaborn jupyter nbformat nbclient
python3 scripts/download_datasets.py --data-root data/raw --log-path reports/dataset_download_log.json
.venv/bin/python scripts/build_checkpoint_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/checkpoint1_dataset_selection.ipynb
```

## Notes on Data and Licensing

- GRBench: Apache-2.0 (per HF dataset card).
- ogbn-arxiv: use under OGB/source benchmark terms and required citations.
- SNAP com-Amazon: follow SNAP data usage and citation norms.

## GitHub Portfolio

- Public repository URL: `Pending publication (requires gh auth login)`
- First notebook to highlight in portfolio: `notebooks/checkpoint1_dataset_selection.ipynb`
