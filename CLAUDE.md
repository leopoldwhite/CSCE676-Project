# CLAUDE.md — CSCE 676 Course Project Navigation Guide

## Project Identity

- **Course**: CSCE 676 Data Mining & Analysis, Spring 2026, Texas A&M University
- **Student**: Yuyang Bai (ybai@tamu.edu), Student ID: 637002999
- **Dataset**: GRBench (Graph Reasoning Benchmark) — 1,740 Q&A pairs across 10 domains, 3 difficulty levels
- **GitHub**: https://github.com/leopoldwhite/CSCE676-Project
- **Canvas**: https://canvas.tamu.edu/courses/432271/assignments

---

## Repository Structure

```
CSCE676-Project/
├── CLAUDE.md                                    ← YOU ARE HERE
├── README.md                                    ← Public project documentation
├── .gitignore
├── notebooks/
│   ├── checkpoint1_dataset_selection.ipynb       ← CP1: Dataset selection + EDA (392KB)
│   ├── checkpoint2_rq_formation.ipynb            ← CP2: Research questions + initial methods (99KB)
│   └── checkpoint2_rq_formation.py               ← CP2 source script
├── reports/
│   ├── dataset_download_log.json                 ← Download metadata + SHA256 checksums
│   ├── dataset_profiles.json                     ← Dataset statistics
│   └── figures/
│       ├── grbench_domain_level_counts.png       ← CP1: Domain-level distribution
│       ├── grbench_domain_level_heatmap.png      ← CP1: Domain-difficulty heatmap
│       ├── grbench_length_by_level.png           ← CP1: Question length by difficulty
│       ├── grbench_lexical_patterns.png          ← CP1: Lexical patterns
│       ├── cp2_umap_embeddings.png               ← CP2: UMAP 2D projection (domain vs difficulty)
│       ├── cp2_graph_stats_by_domain.png         ← CP2: Graph stats per domain (nodes/density/clustering/components)
│       ├── cp2_global_centrality.png             ← CP2: Top-15 keywords by PageRank & betweenness
│       ├── cp2_louvain_per_domain.png            ← CP2: Louvain communities per domain
│       ├── cp2_pagerank_vs_difficulty.png         ← CP2: Centrality-difficulty correlation
│       ├── cp2_reasoning_complexity.png          ← CP2: Entity count / causal language / length by difficulty
│       ├── cp2_kmeans_sweep.png                  ← CP2: K-Means silhouette/ARI over k=3..11
│       ├── cp2_distance_distribution.png         ← CP2: Pairwise cosine distance histogram
│       └── cp2_baseline_comparison.png           ← CP2: Difficulty classification baseline (domain-only vs embeddings vs combined)
├── scripts/
│   ├── download_datasets.py                      ← 3-dataset downloader with checksums (175 lines)
│   └── build_checkpoint_notebook.py              ← Notebook generator from templates
└── data/
    └── raw/                                      ← Downloaded datasets (not committed to git)
        └── grbench/                              ← GRBench JSONL files per domain
```

---

## Dataset: GRBench

- **Source**: https://huggingface.co/datasets/PeterJinGo/GRBench
- **License**: Apache-2.0
- **Size**: 1,740 Q&A pairs
- **Domains (10)**: amazon (200), biology (140), chemistry (140), computer_science (150), healthcare (270), legal (180), literature (240), materials_science (140), medicine (140), physics (140)
- **Difficulty levels**: easy (780), medium (920), hard (120)
- **Columns**: domain, qid, question, answer, level
- **Key properties**: Questions range from short factoids (~10 words) to complex multi-sentence prompts (~250 words). Answers mostly short (median ~2 words) with heavy right tail (max 362 words).

---

## Research Questions

### RQ1 (Course — Text Mining + Clustering)
**Can text embeddings and unsupervised clustering reveal latent question types that predict difficulty level better than domain labels alone?**

| Aspect | Detail |
|--------|--------|
| Task type | Text mining + clustering |
| Course techniques | Sentence embeddings, K-Means, DBSCAN, dimensionality reduction |
| Algorithms | Sentence-BERT (all-MiniLM-L6-v2), K-Means, DBSCAN, UMAP |
| Evaluation | Silhouette score, Adjusted Rand Index (ARI), cluster purity, classification F1 |
| Baseline | Domain-only difficulty prediction (Random Forest) |
| Key CP2 finding | UMAP shows overlapping clusters beyond domain boundaries; embeddings outperform domain-only features for difficulty prediction |

### RQ2 (Course — Graph Mining + Community Detection)
**What structural properties of keyword co-occurrence graphs differentiate domains, and do graph centrality measures correlate with question difficulty?**

| Aspect | Detail |
|--------|--------|
| Task type | Graph mining, centrality analysis, community detection |
| Course techniques | Graph construction, PageRank, betweenness centrality, Louvain community detection |
| Algorithms | NetworkX co-occurrence graph, PageRank, betweenness centrality, Louvain modularity |
| Evaluation | Graph density, modularity, centrality distributions, Spearman correlation |
| Baseline | Word frequency analysis (no graph structure) |
| Key CP2 finding | Density/clustering/component counts vary across domains; centrality-difficulty correlation shows weak but positive trend |

### RQ3 (External — Graph-Aware Reinforcement Learning)
**Can graph-aware reinforcement learning improve multi-hop reasoning trajectories on GRBench compared to prompting-only baselines?**

| Aspect | Detail |
|--------|--------|
| Task type | Trajectory optimization for graph-augmented QA |
| External technique | Graph-aware RL (PPO / GRPO) |
| Action space | retrieve_from_graph, reason_over_context, generate_answer |
| Evaluation | Accuracy, trajectory efficiency, reward convergence |
| Baseline | Zero-shot prompting, few-shot prompting |
| Key CP2 finding | Harder questions show higher entity counts, more causal language, longer text → supports multi-hop hypothesis; episode length estimates are tractable |

### RQ-to-Method Mapping Table

| RQ | Course/External | Primary Technique | Algorithm(s) | Evaluation Metric(s) |
|----|----------------|-------------------|-------------|---------------------|
| RQ1 | Course | Text mining + Clustering | Sentence-BERT, K-Means, DBSCAN, UMAP | Silhouette, ARI, cluster purity |
| RQ2 | Course | Graph mining + Community detection | Co-occurrence graph, PageRank, Louvain | Density, modularity, centrality-difficulty correlation |
| RQ3 | External | Graph-aware RL | PPO / GRPO, trajectory reward modeling | Accuracy, trajectory efficiency, reward convergence |

---

## Project Timeline & Deliverables

| # | Deliverable | Due Date | Points | Status | What to Submit |
|---|-------------|----------|--------|--------|----------------|
| 1 | **Checkpoint 1**: Dataset Selection + EDA | Feb 12 | 50 | ✅ 50/50 | Fully run .ipynb with A-F sections |
| 2 | **Checkpoint 2**: Research Question Formation | Mar 17 | 50 | ✅ Submitted (pending grade) | Fully run .ipynb with RQs, method mapping, initial runs |
| 3 | **2-Minute Video** | Apr 20 | 100 | ❌ Not started | YouTube link (strict 2:00 ± 2sec) |
| 4 | **AI Phone Interview** | Apr 24 | 150 | ❌ Not started | Call AI hotline, 10-15 min discussion |
| 5 | **Final Deliverable** | Apr 27 | 150 | ❌ Not started | Curated .ipynb + code repo (story, not trials) |
| 6 | **Video Feedback** | Apr 27 | 50 | ❌ Not started | Watch & give feedback on peer videos |

**Total**: 500 pts

---

## Remaining Deliverables — Detailed Requirements

### 2-Minute Video (100 pts, Due Apr 20)

- **Strict 2 minutes ± 2 seconds** — not 1:50, not 2:10
- Post to YouTube, submit link
- Content: motivation → big idea/question → big takeaway
- Framing: "convince an investor to put $2M into your project"
- No deep technical discussion — that's for the Final Deliverable
- Style is flexible (slides + voiceover, etc.)

### AI Phone Interview (150 pts, Due Apr 24)

- Call a special project hotline for 10-15 min AI-driven discussion
- AI Hotline prompt will be provided (so you'll know question types)
- Discussion is transcribed and graded from transcript
- Can call multiple times — only last call is graded
- Essentially an oral exam on project understanding

### Final Deliverable (150 pts, Due Apr 27)

- Final curated notebook + code repository
- Notebook must **tell the story** of the project: motivation → question → results → analysis
- **Will be penalized** for messy/shaggy notebooks with trial-and-error artifacts
- Must be a **coherent, clean narrative**
- Use plenty of markdown cells to guide the reader — "hold our hand and tell us the story"
- Should include deep analysis, findings, and conclusions

### Video Feedback (50 pts, Due Apr 27)

- Watch peer videos and provide feedback
- Videos will be compiled by the instructor and shared to the class

---

## Grading Rubric (applies to CP1, CP2, and likely Final Deliverable)

- **Strong/Professional (full marks)**: Correct and complete; reasonable justified assumptions; thoughtful handling of data issues; clear explanations of what and why; clean readable code passing professional review; meaningful tests.
- **Partial/Developing (half marks)**: Mostly complete with gaps; shallow reasoning; messy but functional code; superficial tests.
- **Minimal/Incorrect (zero)**: Largely incorrect or missing; no reasoning; code doesn't run; no tests.

**Universal requirements for every notebook submission**:
1. Every algorithmic decision must document **WHY**
2. Code must be professional, well-explained, well-documented
3. Collaboration declaration required: (1) Collaborators, (2) Web Sources, (3) AI Tools, (4) Citations

---

## Key Technical Dependencies

From checkpoint notebooks:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sentence-transformers` (all-MiniLM-L6-v2 model)
- `umap-learn`
- `scikit-learn` (KMeans, DBSCAN, RandomForestClassifier, StratifiedKFold)
- `networkx`
- `python-louvain` (community detection)
- `scipy` (Spearman correlation)

---

## Working Conventions

- Python notebooks in `notebooks/`, scripts in `scripts/`, outputs in `reports/`
- Raw data in `data/raw/` (not committed to git)
- Figures saved to `reports/figures/` at 180 DPI
- All code uses type hints (`from __future__ import annotations`)
- Seaborn whitegrid theme, "talk" context for all plots
- Decision protocol: every technical choice documents WHY (4 dimensions: interesting, appropriate, feasible, evaluable)
