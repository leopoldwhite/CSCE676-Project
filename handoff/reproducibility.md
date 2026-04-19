# Reproducibility — CSCE 676 Project Handoff

## Source commits (frozen)

| Repo | Commit | Date | Subject |
|---|---|---|---|
| `CSCE676-Project` | `9580da40e9426c9ca5ef9b1cab16153b1a5bc22b` | 2026-04-10 | Merge pull request #1 from leopoldwhite/master |
| `curriculum_rl_resubmit` | `b16ccc8f40e2a92f25fdd676de135cb4ec978dea` | 2026-04-08 | W1 GRPO baseline: optimize for H100 headroom |

`curriculum_rl_resubmit` is the internal working tree for the paper codebase + artifacts; its public, name-stripped analogue in the course repo is `CSCE676-Project`. Cowork must rename "Curriculum RL (ours)"/"curriculum_rl" to "graph-aware RL" before any output reaches the public repo.

## Dataset snapshot

- **GRBench** @ HuggingFace `PeterJinGo/GRBench` — downloaded locally during CP1 (see `CSCE676-Project/reports/dataset_download_log.json` for SHA256 checksums and download timestamp).
- 1,740 Q&A pairs across 10 domains × 3 difficulty levels.
- Domains: amazon (200), biology (140), chemistry (140), computer_science (150), healthcare (270), legal (180), literature (240), materials_science (140), medicine (140), physics (140).
- Difficulty distribution: easy (700), medium (920), hard (120). Source: `CSCE676-Project/reports/dataset_profiles.json`.
- For RQ3 the scope narrows to the 4 OOD test domains the paper evaluates on: biomedical (270) = healthcare mapped, goodreads (240) = literature mapped, amazon (200), legal (180) → 890 samples.

## Execution mode

**Analysis-only.** No retraining, no inference, no GRBench graph download in this handoff. All RQ3 numbers are extracted from pre-existing artifacts in `curriculum_rl_resubmit/`:

- `Graph-CoT-vllm/evaluation/metrics/*.json` (215 files) and `evaluation/metrics_graph_aware/*.json` (167 files) — per-domain EM / BLEU / ROUGE for zero-shot baseline, prompting-only references (GPT-4o-mini variants, GPT-5 family), curriculum_rl uniform (150 steps) and curriculum_rl curriculum (200 steps).
- `Graph-CoT-vllm/evaluation_gpt4score/` — GPT-4-as-judge scores for the same runs.
- `Graph-CoT-vllm/results_analysis/{ours, vanilla, qwen25_3b-instruct}/{amazon, biomedical, goodreads, legal}/` — raw per-example JSONLs for the three core conditions.
- `grpo-curriculum-h100-w2-mix-125steps.log` (85 MB), `grpo-uniform-h100-w1-baseline-200steps-opt.log` (130 MB), `grpo-uniform-h100-w1-rerun-200steps-sweep.log` (133 MB) — training logs parsed offline for reward / loss / KL curves.
- `curriculum_rl_review_package.md` (1,373 lines) — paper LaTeX, three reviewer reports, author rebuttals (including new-experiment tables), AC/SAC meta review, final decision. Primary source of truth for paper-level claim numbers.

RQ1 and RQ2 re-run the CP2 analyses on the same 1,740-row GRBench snapshot. All seeds pinned: `np.random.seed(42)`, sklearn `random_state=42`, `umap.UMAP(random_state=42)`.

## Hardware (for reference — not used for new experiments)

- **CPU node used for this handoff**: 128-core x86_64, 566 GiB RAM, Ubuntu 24.04 (Linux 6.17.0, glibc 2.39), Python 3.12.3.
- **Original training hardware (from filenames)**: 4× NVIDIA H100 NVL (96 GB each), driver 595.58.03. Training runs consumed 1-2 nodes for ~8-20 hours each (see per-log timestamps).
- **This handoff's compute footprint**: CPU-only; RQ1 embeddings cache from CP2 is reused when possible.

## Wall-clock budget per phase

Recorded at the end of each notebook as a closing cell.

| Phase | Target | Actual |
|---|---|---|
| RQ1 text clustering | ≤30 min CPU | _to fill_ |
| RQ2 graph mining | ≤30 min CPU | _to fill_ |
| RQ3a metric JSON aggregation | ≤5 min | _to fill_ |
| RQ3b reward-curve log parsing | ≤15 min streaming parse | _to fill_ |
| RQ3c trajectory stats + qualitative | ≤10 min | _to fill_ |
| RQ3d paper/rebuttal mining | ≤20 min (manual read + extract) | _to fill_ |
| RQ3e notebook assembly | ≤15 min | _to fill_ |
| Phase 4 handoff packaging | ≤10 min | _to fill_ |

## Seeds

- NumPy: `42`
- scikit-learn: `random_state=42`
- UMAP: `random_state=42`
- PyTorch / CUDA: not exercised in this handoff (analysis-only).

## What is NOT in `handoff/`

- Raw GRBench data (`data/raw/`) — re-download via `CSCE676-Project/scripts/download_datasets.py`; checksums in `reports/dataset_download_log.json`.
- Curriculum RL (ours) model weights (6.4 GB × ~20 checkpoints).
- Full training logs (349 MB combined) — only parsed summaries go to `rq3_graph_aware_rl/figures/` and `metrics.json`.
- The rebuttal package — it lives in `curriculum_rl_resubmit/curriculum_rl_review_package.md` and is summarized in `rq3_graph_aware_rl/notes.md`.
