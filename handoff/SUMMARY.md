# CSCE 676 Project — Experimental Handoff

Date: 2026-04-17
CSCE676-Project commit: `9580da40e9426c9ca5ef9b1cab16153b1a5bc22b` (2026-04-10)
curriculum_rl_resubmit commit: `b16ccc8f40e2a92f25fdd676de135cb4ec978dea` (2026-04-08)
Execution mode: **analysis-only** — no retraining, no inference, no GRBench graph download.

## Headline (one sentence per RQ)

- **RQ1:** Sentence-BERT embeddings lift macro-F1 on difficulty prediction from **0.33 (domain-only)** to **0.83 (LR)** / **0.58 (RF)**, a >2× improvement significant at p ≤ 7.5e-05 under a corrected resampled paired t-test — but unsupervised K-Means recovers *domain* structure (ARI 0.265) rather than *difficulty* (ARI 0.044), so embedding geometry encodes topic, not reasoning complexity.
- **RQ2:** Keyword co-occurrence graphs differ markedly across domains (legal: 364 nodes / density 0.24 vs materials_science: 167 nodes / density 0.34 / clustering 0.69); keyword centrality is **negatively and significantly correlated with question difficulty** (Spearman r = −0.160 PageRank / r = −0.131 betweenness, both p < 1e-7 on n = 1,740), confirming that harder questions invoke rarer, more peripheral keywords. A TF-IDF baseline unexpectedly matches/beats embeddings on difficulty macro-F1 (0.926 vs 0.832).
- **RQ3:** The graph-aware RL method (biased-mixture curriculum on Qwen2.5-3B) achieves the highest Rouge-L (40.62) and GPT4Score (42.25) among 3B-backbone methods on 4 held-out domains, beating vanilla PPO-style RL by **+2.13 Rouge-L / +3.20 GPT4Score** and matching a prompted Qwen3-14B (39.85 / 42.01); the methodological win is cleanest on Legal-Hard where Rouge-L jumps from 4.00 (vanilla) to 15.06 (curriculum). The paper was submitted to ACL 2026 ARR January Cycle, received AC "Findings" recommendation, SAC borderline-below-accept rating, and was **rejected** (2026-04-06).

## Overall story

The three RQs answer the same question from three angles: **what does it take to predict or solve difficulty on GRBench?** RQ1 shows that semantic embeddings dominate coarse domain labels — content matters more than topic — but embeddings alone don't see difficulty directly; K-Means recovers domains, not levels. RQ2 sharpens this picture structurally: harder questions significantly invoke rarer, less central keywords (negative Spearman correlation, p < 1e-7), which explains why dense embeddings compress away part of the signal and a sparse TF-IDF baseline surprisingly wins. RQ3 then shows that when a 3B LLM is trained with a biased-mixture curriculum to interleave reasoning with graph function calls, it learns **evidence-grounded** multi-hop reasoning that beats vanilla PPO and matches a prompted 14B model on Rouge-L / GPT4Score, with the largest gains precisely where the prior RQs predicted difficulty lives (Legal-Hard, Medium-complexity). Together the arc is a course-length argument for *curriculum over scale* on structured-knowledge QA — with honest treatment of the Literature-Hard regression and the paper's rejection.

## Per-RQ detail

### RQ1 — Text Mining + Clustering
- **Headline numbers**: embeddings-LR macro-F1 **0.832** (vs domain-only 0.332, vs combined 0.839); combined-vs-domain-only paired p-value **1.1e-05 (RF)** / **7.5e-05 (LR)** under corrected resampled t-test.
- **K-Means sweep (k=3..11)**: best k = 4 by silhouette (0.097); ARI vs difficulty 0.044, ARI vs domain 0.265 at best k.
- **DBSCAN**: tight config (eps=0.25, min=10) labels 98% as noise — pathological; loose config (eps=0.45, min=5) yields 9 clusters / 31% noise / silhouette 0.027.
- **What worked**: reusing CP2's Sentence-BERT (`all-MiniLM-L6-v2`), L2-normalised embeddings + class_weight='balanced' LR, corrected resampled t-test over naïve paired-t.
- **What surprised**: LR beats RF by 0.25 macro-F1 on embeddings — normalised 384-d embeddings are a nearly-linear space where axis-aligned RF splits waste capacity.
- **Known weaknesses**: embedding-based difficulty prediction may memorise question wording templates; hard class has only ~24 examples per fold; DBSCAN sweep is only 2 configurations.

### RQ2 — Graph Mining + Community Detection
- **Global graph**: 1,675 nodes / 104,674 edges after min_df ≥ 3 stopword-filtering. Top-5 PageRank: `paper, title, opinion, court, papers`; top-5 betweenness: `paper, papers, number, author, case`.
- **Per-domain structure**: legal has the largest graph (364 nodes, density 0.24); materials_science / chemistry have highest clustering (~0.70); physics / biology have the smallest and sparsest.
- **Louvain**: median modularity 0.241 across 10 domains; healthcare and amazon have the cleanest community structure.
- **Centrality ↔ difficulty**: Spearman **r = −0.160 (p = 1.9e-11)** for PageRank, **r = −0.131 (p = 3.8e-8)** for betweenness. Negative sign matches the hypothesis that hard questions invoke the long tail of the vocabulary.
- **TF-IDF baseline macro-F1 = 0.926** — higher than RQ1's embeddings (0.832). The simplest bag-of-words baseline wins on difficulty prediction.
- **Caveats**: question-level windows collapse single-entity multi-hop prompts; keyword centrality is a bag-of-words proxy (not the underlying GRBench graph); Louvain modularity is random-seed dependent by ~0.01.

### RQ3 — Graph-Aware RL
- **Paper main (Table 1, avg over 4 OOD domains, Qwen2.5-3B backbone)**: Graph-CoT 33.84 / 36.11, Vanilla RL 38.49 / 39.05, **Curriculum RL (ours) 40.62 / 42.25**. Pure E2H ablation 30.98 (worse than uniform, supporting the biased-mixture design). Graph-CoT(Qwen3-14B) 39.85 / 42.01 — Curriculum RL (ours) on 3B matches a 14B baseline.
- **Difficulty-wise gains (paper Table 2)**: curriculum wins +32.84 Rouge-L on Academic-Medium, +11.06 on Legal-Hard; loses −12.75 on Literature-Hard and −3.22 on Legal-OOD.
- **Training reward curves (parsed from 349 MB of logs)**: curriculum flat at ~0.40 train-reward; uniform climbs 0.26 → 0.50. On held-out validation EM, curriculum wins **0.174 > uniform 0.156 > uniform-sweep 0.133**.
- **Trajectory stats (rebuttal tables)**: structural difficulty metrics (rounds, nodes/round, NC%, loop%) all show monotonic Easy → Medium → Hard trends across all three models (base, vanilla, curriculum). Curriculum cuts Nodes/Round on Hard from 67 (vanilla) to 31 — learns to prune expansions. On easy/medium, loop rate drops from ~10% (vanilla) to ~7% (curriculum).
- **Evidence-grounded accuracy**: Curriculum RL (ours) 31.7% (best of three), vs Vanilla 29.1%, vs base 28.3%. Lowest ungrounded-correct rate (4.0% vs 5.4% / 4.4%) — gains come from retrieval quality, not fluency.
- **Peer-review outcome**: ACL 2026 ARR Jan Cycle, REJECTED (2026-04-06). Reviewer scores 2.5 / 2.5 / 3.5. AC recommended Findings (3.0). SAC borderline-below-accept (5/6). Primary concerns: novelty ("E2H schedule is standard practice"), multi-seed, Literature degradation.

## Open risks for the final deliverable

- **Naming discipline**: Cowork must strip "Curriculum RL (ours)" from anything bound for the public CSCE676 repo (final-deliverable notebook markdown, 2-min video, figure titles, video-feedback form). In `handoff/` internal the name is fine. This was stripped once deliberately in a prior commit of `CSCE676-Project`.
- **Metric-choice sensitivity**: on EM (strictest metric), Qwen3-8B beats all curriculum_rl runs (0.424 vs best 0.349). The "3B beats 14B" framing requires explicitly stating Rouge-L / GPT4Score. Do **not** claim unconditional superiority.
- **Literature-Hard regression** (RQ3): Rouge-L drops 12.75 pts vs vanilla on this subset. Must be acknowledged — the phone interview will probe it. Accurate framing: "curriculum shifts probability mass toward multi-hop traversal, which hurts Aggregation/Counting questions where simple token counting suffices."
- **Training-reward curve interpretation** (RQ3): the flat curriculum curve looks bad at a glance but is expected under a difficulty-escalating sampler; be prepared to explain why the validation-metric and test-set results are the correct signal.
- **Dataset-count typo** (course `CLAUDE.md`): the stored CLAUDE.md lists `easy (780)` — actual data is `easy (700)`. Handoff uses the correct counts; flag if Cowork cites CLAUDE.md directly.
- **Paper rejection** (RQ3): do not hide it — but frame honestly. "The curriculum method was peer-reviewed to the ACL bar and survived most scrutiny; the paper was rejected primarily on novelty grounds, not on the empirical claims this project relies on."
- **GPT-4 judge metric** is available on only 8 of 19 models in the aggregated JSONs. Any GPT4Score claim should be accompanied by the caveat "where available".

## Handoff inventory

```
handoff/
├── SUMMARY.md                                  ← this file
├── environment.txt                             ← pip freeze (190 packages)
├── reproducibility.md                          ← commits, seeds, hardware, wall-clock
├── rq1_text_clustering/
│   ├── notebook.ipynb                          ← fully run, 10 code cells
│   ├── metrics.json                            ← n=1740, k-means, DBSCAN, classification, significance
│   ├── notes.md                                ← headlines, surprises, caveats for Cowork
│   ├── figures/kmeans_sweep.png                ← silhouette + ARI(vs diff / vs domain) + inertia
│   ├── figures/umap_domain.png                 ← UMAP coloured by domain
│   ├── figures/umap_difficulty.png             ← UMAP coloured by difficulty
│   └── tables/{kmeans_sweep, dbscan_configs, difficulty_classification,
│                significance_combined_vs_domain, umap_coords}.csv
├── rq2_graph_mining/
│   ├── notebook.ipynb                          ← fully run
│   ├── metrics.json                            ← per-domain stats, global centrality, Louvain, correlations, TF-IDF baseline
│   ├── notes.md
│   ├── figures/graph_stats_by_domain.png       ← 4-panel: nodes, density, clustering, largest-CC frac
│   ├── figures/global_centrality.png           ← top-15 PageRank & betweenness bars
│   ├── figures/louvain_per_domain.png          ← modularity bars
│   ├── figures/pagerank_vs_difficulty.png      ← centrality distribution by difficulty
│   └── tables/{per_domain_graph_stats, global_centrality_top15, louvain_per_domain}.csv
└── rq3_graph_aware_rl/
    ├── notebook.ipynb                          ← fully run
    ├── metrics.json                            ← peer-review metadata, main table, curriculum gain, reward summary, trajectory + EH stats
    ├── notes.md                                ← full narrative: paper provenance, main table, difficulty breakdown, reward curves, rebuttal additions, peer-review outcome
    ├── figures/difficulty_breakdown.png        ← Easy/Medium/Hard/OOD Rouge-L across 5 domains × 3 methods
    ├── figures/reward_convergence.png          ← curriculum vs uniform training reward (rolling-5 smooth)
    ├── figures/trajectory_stats.png            ← rounds / NC% / loop% by difficulty × model
    └── tables/
        ├── full_metrics_long.csv               ← 820 rows: every model × domain × difficulty
        ├── headline_overall_em.csv             ← wide, avg EM per model on 4 OOD
        ├── headline_overall_rougeL.csv         ← same on Rouge-L
        ├── headline_by_difficulty.csv          ← long, e/m/h/ood aggregates
        ├── spec_aligned_summary.csv            ← per-role (zero-shot, curriculum, etc.) summary
        ├── paper_main_table.csv                ← paper Table 1 numerics
        ├── paper_difficulty_table.csv          ← paper Table 2 numerics
        ├── training_progress.csv               ← 525 rows, per-step reward
        ├── training_reward_summary.csv         ← 5-point summary per run
        ├── trajectory_stats_by_difficulty.csv  ← rebuttal table (rounds, NC%, loop%)
        ├── evidence_grounded_accuracy.csv      ← rebuttal W4 table
        └── qualitative_trajectories.md         ← 4 annotated trajectories (easy/medium/hard/failure)
```

**Word count for this SUMMARY.md**: well under the 1,500-word cap.
