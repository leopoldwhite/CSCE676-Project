# Findings

## Candidate Dataset Summary

### 1) GRBench
- Source: https://huggingface.co/datasets/PeterJinGo/GRBench
- Local files are JSONL by domain (not one JSON array per file).
- Parsed local stats:
  - total samples: 1,740
  - domains: 10
  - levels: easy/medium/hard
- Why relevant:
  - Course techniques: graph/text mining and structured reasoning analysis
  - Beyond-course: graph-aware RL for tool-use trajectories
- License/usage:
  - Apache-2.0 (HF dataset card)

### 2) ogbn-arxiv
- Source docs: https://ogb.stanford.edu/docs/nodeprop/
- Downloaded from: https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
- Parsed local stats:
  - nodes: 169,343
  - edges: 1,166,243
  - feature dimension: 128
- Why relevant:
  - Course: graph mining + node classification
  - Beyond-course: GNNs/Graph Transformers/self-supervised graph learning

### 3) SNAP com-Amazon
- Source: https://snap.stanford.edu/data/com-Amazon.html
- Downloaded from: https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
- Parsed local stats:
  - nodes: 334,863
  - edges: 925,872
- Why relevant:
  - Course: graph statistics, centrality, community detection
  - Beyond-course: node embeddings and scalable link prediction

## EDA Findings for Selected Dataset (GRBench)
- Domain size imbalance exists (max/min ratio > 1.9).
- Difficulty distribution is skewed toward medium; hard subset is limited.
- Question and answer lengths are heavy-tailed; long outliers exist.
- No missing values and no duplicate `(domain, qid)` after cleaning.
- Repeated question templates suggest potential lexical/template bias.

## Source Links Used in Notebook
- https://huggingface.co/datasets/PeterJinGo/GRBench
- https://huggingface.co/api/datasets/PeterJinGo/GRBench
- https://ogb.stanford.edu/docs/nodeprop/
- https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
- https://snap.stanford.edu/data/com-Amazon.html
- https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
