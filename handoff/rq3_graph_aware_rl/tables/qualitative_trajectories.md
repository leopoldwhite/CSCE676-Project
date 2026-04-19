# Qualitative trajectories — Curriculum RL (ours) curriculum (w2, step 125)

All examples are sampled from training-time episode logs of the `grpo-curriculum-h100-w2-mix-125steps` run, which corresponds to the **biased-mixture curriculum** method. Rewards are the rule-based reward used by verl's GRPO loop: `score = 1.0` means the extracted answer exactly matches the gold answer; partial credit is awarded for valid format / successful retrieval.

The log format is the verbatim `<think> … </think> <graph> … </graph> <information> … </information> … <answer> … </answer>` trace emitted by the policy inside the GRBench environment.

---

## Easy — 1–2 rounds, success (reward = 1.0)

```text
Golden answers: DSK Supinfocom, Pune
Extracted answer: DSK Supinfocom, Pune
Solution string: <think>The question is asking for the organization of a researcher named Nikita Deshpande. First, I need to find the node corresponding to Nikita Deshpande in the graph.</think><graph>RetrieveNode[Nikita Deshpande]</graph>

<information>The ID of this retrieval target node is 53f474d7dabfaee0d9c6706d.</information>

<think>The question is asking for the affiliation (organization) of Nikita Deshpande. Once we have the node, we need to check its organization feature.</think><graph>NodeFeature[53f474d7dabfaee0d9c6706d, organization]</graph>

<information>The organization feature of 53f474d7dabfaee0d9c6706d are: DSK Supinfocom, Pune.</information>

<think>The affiliation of Nikita Deshpande is DSK Supinfocom, Pune.</think><answer>DSK Supinfocom, Pune</answer>
Format message: Valid sequence format
Reward: score 1.0
--------------------------------
```

## Medium — 3–4 rounds, success (reward = 1.0)

```text
Golden answers: AFIPS National Computer Conference
Extracted answer: AFIPS National Computer Conference
Solution string: <think>The question is asking for the venue where a specific paper ("Private sector needs for trusted/secure computer systems") was published. I need to first find the node corresponding to this paper in the graph.</think>
<graph>RetrieveNode[Private sector needs for trusted/secure computer systems]</graph>

<information>The ID of this retrieval target node is 53e9ac0cb7602d97035ce35b.</information>

<think>The question is asking for the publication venue of the paper. I need to check the node's venue neighbor to find the publication location.</think>
<graph>NeighbourCheck[53e9ac0cb7602d97035ce35b, venue]</graph>

<information>The venue neighbors of 53e9ac0cb7602d97035ce35b are: ['pub_1772'].</information>

<think>The paper was published in the venue with the ID 'pub_1772'. I need to find the name of this venue.</think>
<graph>NodeFeature[pub_1772, name]</graph>

<information>The name feature of pub_1772 are: AFIPS National Computer Conference.</information>

<think>The publication venue of the paper "Private sector needs for trusted/secure computer systems" is AFIPS National Computer Conference.</think>
<answer>AFIPS National Computer Conference</answer>
Format message: Valid sequence format
Reward: score 1.0
--------------------------------
```

## Hard — 5+ rounds, success (reward = 1.0)

```text
Golden answers: Ulrike Wais
Extracted answer: Ulrike Wais
Solution string: <think>The question is asking for the authors of two specific papers. I need to find the nodes for these papers first.</think>
<graph>RetrieveNode[nanoformulation and encapsulation approaches for poorly water soluble drug nanoparticles]</graph>

<information>The ID of this retrieval target node is 2195947172.</information>

<think>The question is asking for the authors of the first paper. I need to find the authors of the node with the ID 2195947172.</think>
<graph>NeighbourCheck[2195947172, author]</graph>

<information>The author neighbors of 2195947172 are: ['2164330685', '2414629550', '2139951898', '2747459195'].</information>

<think>The question is asking for the authors of the first paper. I need to find the authors of the node with the ID 2414629550.</think>
<graph>NodeFeature[2414629550, name]</graph>

<information>The name feature of 2414629550 are: Ulrike Wais.</information>

<think>The question is asking for the authors of the second paper. I need to find the node for this paper first.</think>
<graph>RetrieveNode[highly stable and conductive microcapsules for enhancement of joule heating performance]</graph>

<information>The ID of this retrieval target node is 2414617100.</information>

<think>The question is asking for the authors of the second paper. I need to find the authors of the node with the ID 2414617100.</think>
<graph>NeighbourCheck[2414617100, author]</graph>

<information>The author neighbors of 2414617100 are: ['2634464087', '2106841144', '2414629550', '195369621', '2106437761', '2162888239', '2431054325', '2662780652', '2151082482', '2176129614', '2131578525'].</information>

<think>The question is asking for the authors of the second paper. I need to find the authors of the node with the ID 2414629550.</think>
<graph>NodeFeature[2414629550, name]</graph>

<information>The name feature of 2414629550 are: Ulrike Wais.</information>

<think>Now I have the authors of both papers. They are both Ulrike Wais.</think>
<answer>Ulrike Wais</answer>
Format message: Valid sequence format
Reward: score 1.0
--------------------------------
```

## Failure — stuck in a loop (reward < 0.1)

```text
Golden answers: 26
Extracted answer: Could not find the citation count for the paper "transfer for primary angioplasty the importance of time". Please check the feature name and try again.
Solution string: <think>The question asks for the number of citations of a specific paper titled "transfer for primary angioplasty the importance of time". I need to find this paper node in the graph first.</think>
<graph>RetrieveNode[transfer for primary angioplasty the importance of time]</graph>

<information>The ID of this retrieval target node is 2112704811.</information>

<think>The citation count is stored as a feature of the paper node. I need to find this feature to answer the question.</think>
<graph>NodeFeature[2112704811, citation]</graph>

<information>The node or feature name does not exist in the graph. This might be because your given feature name is not correct. Please modify it.</information>

<answer>Could not find the citation count for the paper "transfer for primary angioplasty the importance of time". Please check the feature name and try again.</answer>
Format message: Unexpected tag <answer> in state information
Reward: final_format_score 0
--------------------------------
```
