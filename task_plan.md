# Task Plan: Course Project Checkpoint 1 (GraphDancer-Oriented)

## Goal
Deliver a complete checkpoint submission: identify 3 candidate datasets (including GRBench), compare them, select one dataset, run EDA, and package a professional runnable notebook for GitHub portfolio use.

## Current Phase
Phase 5 in_progress

## Phases

### Phase 1: Dataset Sourcing and Download Setup
- [x] Confirm three candidate datasets and official sources
- [x] Create download scripts and data folders
- [x] Download all candidate datasets
- **Status:** complete

### Phase 2: Comparative Analysis Draft
- [x] Build required comparison table dimensions
- [x] Map course techniques and beyond-course techniques
- [x] Document feasibility, bias, and ethics
- **Status:** complete

### Phase 3: Dataset Selection and EDA
- [x] Select one dataset with explicit trade-offs
- [x] Implement EDA pipeline (basics, cleaning, bias)
- [x] Generate charts and initial findings
- **Status:** complete

### Phase 4: Notebook and Repo Packaging
- [x] Build fully runnable notebook with A-F sections
- [x] Add README and collaboration declaration
- [x] Verify notebook execution end-to-end
- **Status:** complete

### Phase 5: Publish
- [ ] Commit project artifacts
- [ ] Set repo public and push
- [ ] Report result and any blockers
- **Status:** in_progress

## Key Decisions
| Decision | Rationale |
|---|---|
| Include GRBench as mandatory candidate | User requested it explicitly |
| Select GRBench for EDA | Best fit to GraphDancer trajectory and beyond-course graph-aware RL |
| Download ogbn-arxiv and SNAP com-Amazon as additional candidates | Strong course alignment + accessible baselines |
| Keep raw data out of version control via `.gitignore` | Avoid large repository and keep reproducible download workflow |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| GRBench parsing failed with `JSONDecodeError` when treated as one JSON object | 1 | Switched to JSONL line-by-line parser |
| GitHub publishing blocked by missing authentication (`gh auth status` not logged in) | 1 | Prepared publish-ready repo locally; pending user auth |
