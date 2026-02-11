# GraphDancer Course Project - Checkpoint 1

This folder contains a full submission-ready implementation for Checkpoint 1:
- (A) Candidate dataset identification (3 datasets, including GRBench)
- (B) Comparative analysis table with required dimensions
- (C) Final dataset selection + trade-offs
- (D) EDA on selected dataset (GRBench)
- (E) Initial insights + research questions
- (F) GitHub portfolio setup artifacts

## Candidate Datasets

1. **GRBench**  
   Source: <https://huggingface.co/datasets/PeterJinGo/GRBench>

2. **ogbn-arxiv**  
   Source: <https://ogb.stanford.edu/docs/nodeprop/>

3. **SNAP com-Amazon**  
   Source: <https://snap.stanford.edu/data/com-Amazon.html>

## Selected Dataset

**GRBench** is selected for this checkpoint because it most directly supports the GraphDancer project direction while still covering course topics and enabling beyond-course techniques (graph-aware RL for tool-using LLM trajectories).

## Repository Structure

- `notebooks/checkpoint1_graphdancer.ipynb` - fully runnable notebook submission
- `scripts/download_datasets.py` - one-command dataset downloader
- `scripts/build_checkpoint_notebook.py` - notebook generator
- `reports/dataset_download_log.json` - download metadata and checksums
- `reports/dataset_profiles.json` - parsed dataset size/shape summary
- `reports/figures/` - EDA figures produced by notebook execution
- `data/raw/` - downloaded raw datasets (local use)

## Quickstart

```bash
cd course_project_checkpoint1
uv venv .venv
uv pip install --python .venv/bin/python pandas matplotlib seaborn jupyter nbformat nbclient
python3 scripts/download_datasets.py --data-root data/raw --log-path reports/dataset_download_log.json
.venv/bin/python scripts/build_checkpoint_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/checkpoint1_graphdancer.ipynb
```

## Notes on Data and Licensing

- GRBench: Apache-2.0 (per HF dataset card).
- ogbn-arxiv: use under OGB/source benchmark terms and required citations.
- SNAP com-Amazon: follow SNAP data usage and citation norms.

## GitHub Portfolio

- Public repository URL: `Pending publication (requires gh auth login)`
- First notebook to highlight in portfolio: `notebooks/checkpoint1_graphdancer.ipynb`
