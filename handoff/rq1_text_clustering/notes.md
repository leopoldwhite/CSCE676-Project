# RQ1 — notes for Cowork

## Headline numbers (for SUMMARY.md)

- Embeddings-based difficulty prediction **significantly** beats domain-only baseline on both classifiers: macro-F1 rises from **0.33 → 0.83** for Logistic Regression and **0.33 → 0.58** for Random Forest.
- Adding domain features on top of embeddings is a **rounding error** (RF: +0.0003, LR: +0.008), so the content signal is carried entirely by the text, not by domain membership.
- Unsupervised K-Means does not recover difficulty structure directly: best silhouette is at **k = 4** with silhouette **0.097**, ARI vs difficulty **0.044**, ARI vs domain **0.265**. K-Means finds domain-ish structure, not difficulty.
- Corrected resampled paired t-test (Nadeau & Bengio) on per-fold macro-F1: combined vs domain-only significant at **p = 1.1e-05 (RF)** and **p = 7.5e-05 (LR)**.

## What worked

- **Reusing the CP2 embedding model (`all-MiniLM-L6-v2`)** — direct comparability, cached locally, 0.8 s to encode 1,740 questions, cached as `.npz` so the notebook re-run is near-instant.
- **UMAP + dual colouring (domain vs difficulty)** — the two coloured projections are side-by-side evidence that difficulty sits *inside* domain clusters, not across them. This is the clearest single visual for the 2-minute video.
- **L2-normalised embeddings + LR with `class_weight='balanced'`** — handles the easy/medium/hard class imbalance (700/920/120) cleanly; without rebalancing, LR collapses to the majority class and F1 drops to ~0.3.
- **Corrected resampled t-test over plain paired-t** — plain paired-t is anti-conservative on k-fold CV (the training sets overlap). The Nadeau-Bengio correction is the textbook fix.

## What surprised

- **Logistic Regression beats Random Forest by 0.25 macro-F1** on embeddings. Normalised 384-d embeddings are a nearly-linear space where margin-style classifiers dominate; RF's axis-aligned splits waste capacity. Worth flagging in the video: "the right classifier matters as much as the right features."
- **K-Means ARI vs domain (0.26) >> ARI vs difficulty (0.04)**. Semantic embeddings latch onto *topic* first, not reasoning complexity. This actually motivates RQ2's graph-mining angle — if domain structure is what embeddings see, maybe structural/graph features are what reveal difficulty.
- **DBSCAN tight config is pathological** (98% noise). Reported honestly anyway because the spec required ≥2 configs and the failure mode is informative: cosine distances between question embeddings are rarely below 0.25, so density-based clustering struggles at high thresholds.

## Known weaknesses / caveats

- **Embedding-based difficulty prediction risks memorising wording patterns**, not reasoning difficulty. A question like "How many co-authors has X?" is lexically distinctive for a difficulty level. A more rigorous test would hold out entire question *templates*, not just random folds — the current result may overstate generalisation to genuinely novel question types.
- The **hard** class has only 120 examples (~24 per test fold in 5-fold). F1-hard per class (see `metrics.json.difficulty_f1.*.per_class.hard`) should be cited with this caveat; a single misclassified fold swings the class score by ~4 pts.
- DBSCAN sweep is only 2 configurations. A proper eps-sweep would be stronger but the spec only asked for ≥2.
- `level_counts` in the course `CLAUDE.md` (easy=780) disagrees with the actual data (easy=700). We used the data; flagged in `reproducibility.md`.

## For Cowork (video / interview)

- **Video punchline**: one sentence on embeddings beating domain labels, one sentence on "but clustering alone doesn't see difficulty — it sees topic", and the UMAP figure (`figures/umap_difficulty.png`) as the hero visual.
- **Phone interview likely probes**:
  1. Why Sentence-BERT specifically? → same as CP1/CP2, small, fast, strong baseline for short texts.
  2. Why macro-F1? → class imbalance; we care about `hard` despite it being 7% of data.
  3. Why corrected t-test? → naive paired-t is anti-conservative on CV.
  4. Why ARI vs silhouette? → silhouette is cluster-quality (unsupervised), ARI is alignment with an external label (supervised reference).
  5. The RF vs LR gap: **embeddings are linearly separable after normalisation, RF wastes capacity on axis-aligned splits**.

## Files produced

- `metrics.json` — full numeric dump per INSTRUCTIONS.md §3 schema.
- `figures/kmeans_sweep.png` — silhouette + ARI + inertia across k ∈ {3..11}.
- `figures/umap_domain.png`, `figures/umap_difficulty.png` — two UMAP colourings.
- `tables/kmeans_sweep.csv`, `tables/dbscan_configs.csv`, `tables/difficulty_classification.csv`, `tables/significance_combined_vs_domain.csv`, `tables/umap_coords.csv`.
- `notebook.ipynb` — fully run, 4 s wall-clock on a cached embedding run.
