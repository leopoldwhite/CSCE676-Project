# RQ2 — notes for Cowork

## Headline numbers (for SUMMARY.md)

- **Global keyword co-occurrence graph**: 1,675 nodes, 104,674 edges from 1,740 questions after stopword/DF filtering.
- **Centrality ↔ difficulty, signed and significant**: PageRank Spearman **r = −0.160 (p = 1.9e-11)**, Betweenness **r = −0.131 (p = 3.8e-8)**. Both negative — harder questions invoke *less* central, more peripheral keywords. This is not a huge effect size, but it is strongly significant at n = 1,740 and is directionally consistent with CP2's "hard questions involve rarer entities" finding.
- **Per-domain structure is heterogeneous**: legal has the biggest graph (364 nodes, density 0.24) — its questions cite many named cases/opinions; materials_science and chemistry have the highest clustering (~0.70) — vocabulary is tightly thematic; physics and biology have the smallest, sparsest graphs.
- **Louvain modularity** is in the 0.12–0.37 band across domains (median 0.24). Domains with higher modularity (e.g. healthcare, amazon) have cleaner sub-topic community structure; domains with lower modularity (chemistry, materials_science) are already so tightly clustered that further partitioning gives diminishing returns.
- **TF-IDF + LR baseline** macro-F1 = **0.926** — higher than RQ1's embeddings-only macro-F1 (0.832). Sparse word-frequency features + rebalanced linear classifier beat 384-d dense embeddings for difficulty prediction.

## What worked

- **Question-level document window** (instead of sentence window) — GRBench questions are typically one sentence, so sentence windowing would add complexity without new signal. Confirmed by the avg tokens/question (~18 after stopwording).
- **min_df ≥ 3 threshold** — removes hapax legomena that otherwise dominate node count. Without this cutoff CP2 had graphs with 2× the nodes but no clearer structure.
- **Weighted PageRank, unweighted betweenness** — weighted betweenness inverts the semantics of co-occurrence weight (high weight should be "close", but shortest-path cost treats it as "near-zero distance"). Using unweighted betweenness for ranking keeps the interpretation clean.
- **Top PageRank words (paper, title, opinion, court, case, author)** are structural: they combine legal (opinion/court/case) and academic (paper/title/author) vocabulary and recur across domains — which is why they surface as central hubs on the union graph. The betweenness list is similar because high-PR words are also structural bridges at this scale.

## What surprised

- **TF-IDF beats embeddings for difficulty (0.926 vs 0.832)** — the simplest baseline wins. Two candidate explanations:
  1. Difficulty correlates strongly with specific word-markers ("how many", "relationship between", "which of", etc.) that TF-IDF sees literally but Sentence-BERT abstracts into vector geometry.
  2. Sentence-BERT L2 normalisation collapses absolute word frequency into direction-only information, giving up a signal TF-IDF retains.
  Either way, the video/interview should **not** frame embeddings as strictly better than bag-of-words here — it's the opposite.
- **Centrality-difficulty correlation is negative and significant**, even though CP2 reported it as "weak but positive" — the sign flips when we (a) apply min_df ≥ 3 and (b) use unweighted betweenness. This is a real methodological discrepancy; CP2 over-counted noise keywords that inflated centrality for hard questions.
- **Legal's graph is ~2× larger than any other domain**. 364 nodes vs ~140 for the typical domain. Cause: legal questions name specific cases and opinions, which survive the DF ≥ 3 cutoff because case names recur in multi-question clusters. Healthcare and literature are similarly content-heavy but less name-dense.

## Known weaknesses / caveats

- **Co-occurrence windows at question level are coarse.** A proper multi-hop question like "A's co-authors who published in B" puts A and B into the same co-occurrence edge, which is the "right" behavior for our hypothesis, but a longer question about a single entity contributes no edges — we are blind to single-entity multi-hop reasoning.
- **Keyword centrality is a bag-of-words proxy** — it sees the words in the question, not the knowledge graph the question is asked over. The "real" centrality analysis would run on the underlying GRBench graph (DBLP / MAG / legal / etc.) which is out of scope for this course project but is explicitly where RQ3 lives.
- **Louvain modularity is random-seed dependent** (the tie-breaking order matters on dense graphs). We fix `random_state=42` but another run can shift modularity by ~0.01. The between-domain *ranking* is stable across seeds in our dev runs.
- **The TF-IDF-beats-embeddings finding depends on class balancing.** Without `class_weight='balanced'`, TF-IDF falls to ~0.45 (majority-class regress) and the gap disappears. This is honest evaluation, but worth flagging in the interview.

## Cross-RQ link (for the final deliverable narrative)

The RQ1 → RQ2 → RQ3 arc now reads:

- **RQ1** — semantic embeddings beat domain labels by ~0.5 macro-F1, showing that *question content* carries difficulty signal beyond the coarse domain taxonomy.
- **RQ2** — but (a) a simpler word-frequency baseline beats embeddings, and (b) the graph we build from co-occurrence captures a weak-but-significant *negative* correlation between keyword centrality and difficulty. Together, these findings suggest that **difficulty is encoded in lexical rarity** as much as in semantic meaning.
- **RQ3** — those rare/peripheral keywords are precisely the ones that require multi-hop graph traversal to ground. Graph-aware RL should, in principle, beat prompting-only models exactly on the subset of questions where keyword centrality is lowest. (The paper/rebuttal mining in Phase 3d will tell us whether the empirical result matches this hypothesis.)

## Files produced

- `metrics.json` — full numeric dump.
- `figures/graph_stats_by_domain.png`, `figures/global_centrality.png`, `figures/louvain_per_domain.png`, `figures/pagerank_vs_difficulty.png`.
- `tables/per_domain_graph_stats.csv`, `tables/global_centrality_top15.csv`, `tables/louvain_per_domain.csv`.
- `notebook.ipynb` — fully run (~15 s wall-clock).
