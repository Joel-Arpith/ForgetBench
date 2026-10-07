# Literature Review

Numbers in brackets refer to [`report/references.md`](../report/references.md).

## 1. Exact unlearning
- **SISA** (Bourtoule et al., 2021) [5] — split data into shards, train one model per shard, aggregate predictions. A deletion retrains only the affected shard: exact, but still retrain-scale.
- **Certified removal** (Guo et al., 2020) [6] — post-deletion model statistically indistinguishable from one never trained on the data; proven for linear models.

## 2. Approximate unlearning for LLMs
- **Gradient ascent** — maximise loss on the forget set; simple, tends to collapse the model.
- **NPO** (Zhang et al., 2024) [7] — preference-style loss that lowers forget-answer likelihood relative to a frozen reference; avoids gradient ascent's collapse.
- **RMU** (Li et al., 2024, WMDP) [8] — steers internal activations on forget inputs towards a random direction.
- **Who's Harry Potter?** (Eldan & Russinovich, 2023) [9] — removed a book series from Llama-2 in ~1 GPU-hour vs 184K GPU-hours of pre-training.

## 3. Benchmarks
- **TOFU** (Maini et al., 2024) [10] — 200 fictitious author profiles; forget quality vs model utility.
- **MUSE** (Shi et al., 2025) [11] — six evaluation dimensions incl. privacy leakage and repeated requests.
- Both score a model **once**, right after unlearning — never after an attack.

## 4. Relearning attacks
- **Jogging the Memory** (Hu et al., 2024) [12] — a light fine-tune on related data revives "forgotten" knowledge.
- **Unlearning isn't deletion** (2025) [13] — most approximate unlearning suppresses rather than removes.

## 5. Robust unlearning
- **SAM perspective** (Fan et al., 2025) [14], building on **SAM** (Foret et al., 2021) [15] — flat minima resist relearning.
- **Random perturbations** (Huu-Tien et al., 2025) [16] — lightweight robustness.

## 6. Privacy measurement
- **Min-K% Prob** (Shi et al., 2023) [17]; MIA on LLMs is often near random (Duan et al., 2024) [18]; comparing before/after models can itself leak (Chen et al., 2021) [19].

## Research gap
| Approach | Cost | Durable under attack | Durability measured? |
|---|---|---|---|
| Full retrain | Very high | Yes | Not needed |
| SISA | High | Yes | No |
| GA / NPO / RMU | Low | Often reversible | No |
| TOFU / MUSE | — | — | Once, not after attack |
| **ForgetBench + HRU** | Low–moderate | Target | **Yes** |
