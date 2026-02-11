#!/usr/bin/env python3
"""Build checkpoint notebook with all required sections (A-F)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()

    cells = []

    cells.append(
        md(
            f"""# Course Project Checkpoint 1: Dataset Selection and EDA

**Student:** [Your Name]  
**Date:** {date.today().isoformat()}  
**Checkpoint scope:** dataset identification, comparative analysis, dataset selection, EDA, initial insights, GitHub portfolio setup.

## Why this notebook exists
I want to turn my graph-reasoning research direction into the semester project. This notebook completes the first checkpoint end-to-end and documents every major algorithmic choice with explicit rationale.
"""
        )
    )

    cells.append(
        md(
            """## Decision Protocol (used throughout)
For each technical choice, I explicitly answer **WHY**:
1. Why this data source and not alternatives.
2. Why this preprocessing is sufficient for a checkpoint-stage EDA.
3. Why this metric/visual helps shape project direction.
4. Why this selected dataset best supports both course and beyond-course techniques.
"""
        )
    )

    cells.append(
        code(
            """from __future__ import annotations

from pathlib import Path
import json
import gzip
import re
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk")
pd.set_option("display.max_colwidth", 180)

# Resolve project root robustly, regardless of where the notebook is executed.
cwd = Path.cwd()
if (cwd / "data" / "raw").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "data" / "raw").exists():
    PROJECT_ROOT = cwd.parent
else:
    raise FileNotFoundError("Could not locate project root containing data/raw")

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
REPORT_ROOT = PROJECT_ROOT / "reports"
FIG_ROOT = REPORT_ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

print("Environment ready. Data and report directories resolved.")
"""
        )
    )

    cells.append(
        md(
            """## (A) Identification of Candidate Datasets

I use three candidates (including GRBench, as required by project intent):
1. **GRBench** (graph-augmented QA benchmark; selected candidate for deep EDA).
2. **ogbn-arxiv** (citation graph benchmark from Open Graph Benchmark).
3. **SNAP com-Amazon** (product co-purchasing graph).

**Why this trio:** all three support core course graph-mining topics, but each also offers a distinct beyond-course path (tool-using LLM agents, GNNs/Graph Transformers, and large-scale graph representation learning).
"""
        )
    )

    cells.append(
        code(
            """profiles = json.loads((REPORT_ROOT / "dataset_profiles.json").read_text(encoding="utf-8"))

candidate_rows = [
    {
        "Dataset": "GRBench",
        "Source": "https://huggingface.co/datasets/PeterJinGo/GRBench",
        "Course Topic Alignment": "Graph mining + text mining over external knowledge graphs",
        "Beyond-Course Technique(s)": "Graph-aware RL for tool-using LLM agents; trajectory-level policy optimization",
        "Dataset Size & Structure": f"{profiles['grbench']['total_samples']} QA samples across {profiles['grbench']['domains']} domain files (JSONL)",
        "Data Types": "domain, qid, question (text), answer (text), level (easy/medium/hard)",
        "Target Variable(s)": "Reference answer text (generative QA target)",
        "Licensing / Usage Constraints": "Apache-2.0 on HF dataset card",
    },
    {
        "Dataset": "ogbn-arxiv",
        "Source": "https://ogb.stanford.edu/docs/nodeprop/",
        "Course Topic Alignment": "Graph analytics, node classification, centrality/community-aware analysis",
        "Beyond-Course Technique(s)": "GNNs (GCN/GraphSAGE/GAT), graph transformers, self-supervised graph pretraining",
        "Dataset Size & Structure": f"{profiles['ogbn_arxiv']['nodes']:,} nodes, {profiles['ogbn_arxiv']['edges']:,} edges, {profiles['ogbn_arxiv']['feature_dim']}-dim node features",
        "Data Types": "edge list, node feature vectors, node year, node labels, temporal split",
        "Target Variable(s)": "arXiv category label (node classification)",
        "Licensing / Usage Constraints": "OGB benchmark terms + source dataset citation requirements",
    },
    {
        "Dataset": "SNAP com-Amazon",
        "Source": "https://snap.stanford.edu/data/com-Amazon.html",
        "Course Topic Alignment": "Graph mining, community discovery, graph statistics",
        "Beyond-Course Technique(s)": "Node2Vec/DeepWalk embeddings, scalable link prediction, graph representation learning",
        "Dataset Size & Structure": f"{profiles['snap_com_amazon']['nodes']:,} nodes, {profiles['snap_com_amazon']['edges']:,} undirected edges (edge list)",
        "Data Types": "from-node ID, to-node ID (unattributed graph)",
        "Target Variable(s)": "No direct label (unsupervised/structural tasks)",
        "Licensing / Usage Constraints": "SNAP data usage/citation norms (research/benchmark usage)",
    },
]

candidate_df = pd.DataFrame(candidate_rows)
candidate_df
"""
        )
    )

    cells.append(
        md(
            """### Candidate Summary Notes
- **GRBench** is the strongest conceptual bridge from my graph-reasoning focus to the class project.
- **ogbn-arxiv** is a robust fallback with mature baselines and reproducible splits.
- **com-Amazon** is computationally feasible while still large enough for realistic graph mining constraints.
"""
        )
    )

    cells.append(
        md(
            """## (B) Comparative Analysis of Datasets

**Algorithmic decision:** I compare along exactly the required rubric dimensions, because these dimensions determine not only feasibility but project risk (data quality, bias, ethics).
"""
        )
    )

    cells.append(
        code(
            """comparison_rows = [
    {
        "Dataset": "GRBench",
        "Supported Data Mining Tasks": "Course: graph/text reasoning over graph-structured knowledge. External: graph-aware RL for tool trajectories.",
        "Data Quality Issues": "Long-tail domain heterogeneity; mixed answer lengths; possible annotation inconsistency across domains.",
        "Algorithmic Feasibility": "High for checkpoint EDA (small-to-medium size); moderate for full RL training.",
        "Bias Considerations": "Domain imbalance (e.g., healthcare/literature larger than some science domains); prompt-style bias.",
        "Ethical Considerations": "Legal/medical QA may produce overconfident incorrect answers if used without human oversight.",
    },
    {
        "Dataset": "ogbn-arxiv",
        "Supported Data Mining Tasks": "Course: graph mining + node classification. External: GNN/Graph Transformer and temporal graph learning.",
        "Data Quality Issues": "Citation graph incompleteness, noisy labels, temporal shift between train/test splits.",
        "Algorithmic Feasibility": "Feasible on a laptop for sampling/baselines; full model sweeps need careful resource control.",
        "Bias Considerations": "Field/time imbalance can bias predictions toward dominant subfields and eras.",
        "Ethical Considerations": "Potentially amplifies historical publication inequities in academic visibility.",
    },
    {
        "Dataset": "SNAP com-Amazon",
        "Supported Data Mining Tasks": "Course: graph statistics, centrality, community detection. External: node embeddings/link prediction.",
        "Data Quality Issues": "No node attributes, no labels, historical snapshot limits temporal validity.",
        "Algorithmic Feasibility": "Highly feasible for classical graph methods; representation learning feasible with mini-batching.",
        "Bias Considerations": "Commercial co-purchase graph can embed popularity and exposure bias.",
        "Ethical Considerations": "Recommendation-style analysis can reinforce feedback loops and market concentration.",
    },
]

comparison_df = pd.DataFrame(comparison_rows)
comparison_df
"""
        )
    )

    cells.append(
        md(
            """## (C) Dataset Selection

**Decision rationale:** use a lightweight weighted score for transparency, then confirm with qualitative trade-off analysis.

Scoring criteria:
- Project-fit to my research direction (weight 0.35)
- Course-technique coverage (0.20)
- Beyond-course novelty (0.25)
- Data access + reproducibility for this semester (0.20)
"""
        )
    )

    cells.append(
        code(
            """weights = {
    "project_fit": 0.35,
    "course_coverage": 0.20,
    "beyond_novelty": 0.25,
    "reproducibility": 0.20,
}

# Scores are 1-5 and intentionally conservative.
score_rows = [
    {"Dataset": "GRBench", "project_fit": 5, "course_coverage": 4, "beyond_novelty": 5, "reproducibility": 4},
    {"Dataset": "ogbn-arxiv", "project_fit": 3, "course_coverage": 5, "beyond_novelty": 4, "reproducibility": 5},
    {"Dataset": "SNAP com-Amazon", "project_fit": 3, "course_coverage": 4, "beyond_novelty": 4, "reproducibility": 5},
]
score_df = pd.DataFrame(score_rows)
score_df["weighted_score"] = sum(score_df[c] * w for c, w in weights.items())
score_df = score_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
score_df
"""
        )
    )

    cells.append(
        md(
            """### Selected Dataset: **GRBench**

**Why selected:**
- Direct continuity with my graph-reasoning research direction.
- Naturally supports course graph/text mining themes.
- Enables at least one clearly beyond-course method: graph-aware RL for multi-round tool-use trajectories.

**Trade-offs accepted:**
- Smaller total sample size than some industrial graph datasets.
- QA-focused supervision means fewer classic node-label tasks.
- Needs careful handling of domain and difficulty imbalance.
"""
        )
    )

    cells.append(
        md(
            """## (D) Exploratory Data Analysis (Selected Dataset Only: GRBench)

**Algorithmic decision:** parse JSONL directly (not schema inference tools) to keep preprocessing deterministic and auditable.
"""
        )
    )

    cells.append(
        md(
            """### Data Collection and Provenance (for EDA)
- Data source: Hugging Face dataset `PeterJinGo/GRBench`.
- Collection process for this project: scripted download via `scripts/download_datasets.py` from official dataset listing/API.
- Provenance record: `reports/dataset_download_log.json` includes source URLs, file sizes, and SHA256 checksums.
- Why this approach: transparent and reproducible local collection; no manual copy-paste steps.
"""
        )
    )

    cells.append(
        code(
            """def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

records = []
for path in sorted((DATA_ROOT / "grbench").glob("*.json")):
    domain = path.stem
    for row in read_jsonl(path):
        records.append(
            {
                "domain": domain,
                "qid": str(row.get("qid", "")).strip(),
                "question": str(row.get("question", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
                "level": str(row.get("level", "")).strip().lower(),
            }
        )

df_raw = pd.DataFrame(records)

# Cleaning choices:
# 1) trim text whitespace,
# 2) standardize level labels,
# 3) remove exact duplicate rows (if any).
expected_levels = {"easy", "medium", "hard"}

df_clean = df_raw.copy()
df_clean["level"] = df_clean["level"].replace({"": "unknown"})
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# Feature engineering for EDA (word lengths are interpretable and robust).
df_clean["question_len_words"] = df_clean["question"].str.split().str.len()
df_clean["answer_len_words"] = df_clean["answer"].str.split().str.len()

# Validation tests (non-trivial behavior checks).
assert df_clean["question"].eq("").sum() == 0, "Empty questions found"
assert df_clean["answer"].eq("").sum() == 0, "Empty answers found"
assert df_clean.duplicated(subset=["domain", "qid"]).sum() == 0, "Duplicate qid within domain found"
assert set(df_clean["level"].unique()).issubset(expected_levels), "Unexpected level labels detected"

print(df_clean.shape)
df_clean.head()
"""
        )
    )

    cells.append(
        code(
            """summary_table = pd.DataFrame(
    {
        "metric": [
            "total_samples",
            "num_domains",
            "num_levels",
            "missing_questions",
            "missing_answers",
            "duplicate_(domain,qid)",
            "avg_question_words",
            "avg_answer_words",
            "max_question_words",
            "max_answer_words",
        ],
        "value": [
            len(df_clean),
            df_clean["domain"].nunique(),
            df_clean["level"].nunique(),
            int(df_clean["question"].isna().sum()),
            int(df_clean["answer"].isna().sum()),
            int(df_clean.duplicated(subset=["domain", "qid"]).sum()),
            round(float(df_clean["question_len_words"].mean()), 2),
            round(float(df_clean["answer_len_words"].mean()), 2),
            int(df_clean["question_len_words"].max()),
            int(df_clean["answer_len_words"].max()),
        ],
    }
)
summary_table
"""
        )
    )

    cells.append(
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=df_clean, x="domain", order=sorted(df_clean["domain"].unique()), ax=axes[0], color="#4C78A8")
axes[0].set_title("GRBench Samples by Domain")
axes[0].set_xlabel("Domain")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=40)

sns.countplot(data=df_clean, x="level", order=["easy", "medium", "hard"], ax=axes[1], palette="Set2")
axes[1].set_title("GRBench Samples by Difficulty Level")
axes[1].set_xlabel("Level")
axes[1].set_ylabel("Count")

fig.tight_layout()
fig.savefig(FIG_ROOT / "grbench_domain_level_counts.png", dpi=180, bbox_inches="tight")
plt.show()
"""
        )
    )

    cells.append(
        code(
            """pivot = pd.crosstab(df_clean["domain"], df_clean["level"]).reindex(columns=["easy", "medium", "hard"], fill_value=0)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title("Difficulty Distribution by Domain")
ax.set_xlabel("Level")
ax.set_ylabel("Domain")
fig.tight_layout()
fig.savefig(FIG_ROOT / "grbench_domain_level_heatmap.png", dpi=180, bbox_inches="tight")
plt.show()
pivot
"""
        )
    )

    cells.append(
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df_clean, x="level", y="question_len_words", order=["easy", "medium", "hard"], ax=axes[0], palette="Set3")
axes[0].set_title("Question Length by Difficulty")
axes[0].set_xlabel("Level")
axes[0].set_ylabel("Words")

sns.boxplot(data=df_clean, x="level", y="answer_len_words", order=["easy", "medium", "hard"], ax=axes[1], palette="Set1")
axes[1].set_title("Answer Length by Difficulty")
axes[1].set_xlabel("Level")
axes[1].set_ylabel("Words")

fig.tight_layout()
fig.savefig(FIG_ROOT / "grbench_length_by_level.png", dpi=180, bbox_inches="tight")
plt.show()

df_clean.groupby("level")[["question_len_words", "answer_len_words"]].mean().round(2)
"""
        )
    )

    cells.append(
        code(
            """# Lightweight lexical bias probe: question openers and keyword concentration.
def first_word(text: str) -> str:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return tokens[0] if tokens else "<empty>"

openers = df_clean["question"].apply(first_word)
opener_counts = openers.value_counts().head(12)

stop = {
    "the", "a", "an", "is", "are", "of", "in", "on", "for", "with", "to", "and", "who",
    "what", "which", "where", "when", "how", "do", "did", "does", "be", "by", "from", "paper", "titled",
}
all_tokens = []
for q in df_clean["question"].str.lower():
    toks = re.findall(r"[a-z']+", q)
    all_tokens.extend([t for t in toks if t not in stop and len(t) > 2])

top_keywords = pd.DataFrame(Counter(all_tokens).most_common(15), columns=["token", "count"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=opener_counts.values, y=opener_counts.index, ax=axes[0], color="#72B7B2")
axes[0].set_title("Most Common Question Openers")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("First Word")

sns.barplot(data=top_keywords, x="count", y="token", ax=axes[1], color="#F58518")
axes[1].set_title("Top Question Keywords (Stopwords Removed)")
axes[1].set_xlabel("Count")
axes[1].set_ylabel("Token")

fig.tight_layout()
fig.savefig(FIG_ROOT / "grbench_lexical_patterns.png", dpi=180, bbox_inches="tight")
plt.show()

opener_counts, top_keywords.head(10)
"""
        )
    )

    cells.append(
        code(
            """# Bias + feasibility diagnostics to motivate next checkpoint methods.
bias_report = pd.DataFrame(
    {
        "metric": [
            "domain_imbalance_ratio(max/min)",
            "medium_level_share",
            "hard_level_share",
            "questions_starting_with_'who'",
            "answers_longer_than_20_words",
        ],
        "value": [
            round(df_clean["domain"].value_counts().max() / df_clean["domain"].value_counts().min(), 2),
            round((df_clean["level"] == "medium").mean(), 3),
            round((df_clean["level"] == "hard").mean(), 3),
            round(df_clean["question"].str.lower().str.contains(r"\\bwho\\b").mean(), 3),
            round((df_clean["answer_len_words"] > 20).mean(), 3),
        ],
    }
)
bias_report
"""
        )
    )

    cells.append(
        md(
            """## (E) Initial Insights and Direction

### Initial observations
1. Domain sizes are imbalanced (notably larger healthcare and literature subsets), so naive aggregate metrics may hide weak-domain behavior.
2. Medium questions dominate, while hard questions are sparse; this can bias model tuning toward medium-difficulty performance.
3. Question and answer length distributions are heavy-tailed, suggesting robust handling of long outputs is necessary.
4. Lexical patterns show recurrent templates, which may inflate performance for prompt-matching strategies.

### Hypotheses for next checkpoint
1. Curriculum-style sampling by difficulty and domain will improve hard-sample robustness versus uniform sampling.
2. Trajectory-level evaluation (tool-call quality, evidence sufficiency) will reveal failure modes hidden by answer-only metrics.

### Potential research questions (RQs)
1. How much does domain-aware reweighting change hard-level accuracy/quality?
2. Does graph-aware RL improve trajectory efficiency versus prompting-only baselines on GRBench?
3. Which error types dominate in legal and healthcare domains under distribution shift?
"""
        )
    )

    cells.append(
        md(
            """## (F) GitHub Portfolio Building

- Public repository (to submit): `Pending publication (run gh auth login, then gh repo create ... --public --push)`
- First notebook: `notebooks/checkpoint1_dataset_selection.ipynb`
- Supporting artifacts:
  - `scripts/download_datasets.py`
  - `reports/dataset_download_log.json`
  - `reports/dataset_profiles.json`
  - `reports/figures/*.png`

### README expectations
- Project motivation and scope.
- Candidate datasets + final selection rationale.
- Reproducible setup steps.
- Results snapshot and next checkpoint plan.
"""
        )
    )

    cells.append(
        md(
            """## Collaboration Declaration

### (1) Collaborators
- None for this checkpoint.

### (2) Web Sources
- GRBench dataset card: https://huggingface.co/datasets/PeterJinGo/GRBench  
- GRBench dataset API metadata: https://huggingface.co/api/datasets/PeterJinGo/GRBench  
- OGB node property prediction docs: https://ogb.stanford.edu/docs/nodeprop/  
- OGB arxiv download endpoint: https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip  
- SNAP com-Amazon page: https://snap.stanford.edu/data/com-Amazon.html  
- SNAP com-Amazon download endpoint: https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz

### (3) AI Tools
- OpenAI Codex (GPT-5-based coding agent) for workflow automation, code generation, and notebook structuring.

### (4) Citations for Papers Used
- Jin, Y., et al. (2024). *Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs*.  
- Bai, Y., et al. (2026). *A Graph-Aware Curriculum for Reinforcement Fine-Tuning in Graph-Augmented Reasoning*.  
- Hu, W., et al. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs*.
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }

    out = Path("notebooks/checkpoint1_dataset_selection.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
