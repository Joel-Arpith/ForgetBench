# ForgetBench — Does Machine Unlearning Survive an Attack?

**Conceptual Project – I · School of Technology, Woxsen University**
Mentor: Prof. Geetha Tripathi

Data-protection law (GDPR Art. 17, India's DPDP Act §12) gives people the right to have their data erased. Deleting a record from a database is easy; removing what a trained language model *learned* from it is not — and approximate unlearning can be undone by a short relearning fine-tune.

**ForgetBench** is a single harness that scores unlearning methods not only on whether a model forgets, but on whether the forgetting **survives a relearning attack** (durability). It compares exact unlearning (full retrain, SISA), approximate unlearning (gradient ascent, NPO, RMU) and our method, **HRU — Hybrid Retention-aware Unlearning**, on identical data, probes, seeds and cost accounting.

## Team

| Member | Roll No. | Role | GitHub | LeetCode | HackerRank |
|---|---|---|---|---|---|
| Bulla Joel Arpith | 25WU0103011 | Harness & experiments | [@Joel-Arpith](https://github.com/Joel-Arpith) | — | — |
| Kanchusthambham Jayanth | 25WU0103021 | Report & presentation | [@kvenkatjayanth89-dotcom](https://github.com/kvenkatjayanth89-dotcom) | — | — |
| Macha Sai Kamal | 25WU0103026 | Data & evaluation | [@Machasaikamal](https://github.com/Machasaikamal) | — | — |
| Shaik Shahul Parvez | 25WU0103036 | Literature & references | [@shahulparvezshaik-stack](https://github.com/shahulparvezshaik-stack) | — | — |
| Damera Sai Sri Ranga Charan | 25WU0102048 | HRU method | [@charandamera18-bit](https://github.com/charandamera18-bit) | — | — |

## Key results (GPT-2 124M, single seed)

| Method | Durability ↓ | Retain log-prob ↑ | Truth ratio (≈1 ideal) | Updates |
|---|---|---|---|---|
| Full retrain | −0.582 | **−0.236** | 1.007 | 150 |
| NPO | −0.762 | −1.594 | 13.166 | 40 |
| Gradient ascent | −1.088 | −1.421 | 1.665 | 40 |
| **HRU (unsharded)** | **−1.309** | −1.692 | 1.292 | 120 |
| HRU (sharded, k=4) | −0.420 | −0.444 | 1.059 | 120 |

- HRU recorded the lowest post-attack forget score of all 12 variants tested.
- **Finding:** SISA-style logit averaging dilutes approximate unlearning — a scrubbed shard at −4.18 reads −0.50 inside a 4-shard ensemble.
- **Self-audit:** durability correlates with model damage (r = +0.88); adjusted for damage, HRU ranks second only to full retraining. A fact-level, equal-budget, 3-seed validation is in `notebooks/07_deciding_experiment.ipynb`.

## Repository layout

| Folder | Contents |
|---|---|
| `notebooks/` | Colab notebooks 01–07 (run in order on a T4 GPU) |
| `src/` | Shared utilities and HRU implementations |
| `eval/`, `data/` | Metrics, relearning attack, synthetic data generation |
| `results/` | Results tables and figures |
| `report/` | Project report and status report |
| `presentation/` | Final deck and speaker script |
| `docs/` | Problem statement, literature review, method notes |

## How to run

Open the notebooks in Google Colab (**Runtime → Change runtime type → T4 GPU**) and run them in numerical order. Each notebook saves to `MyDrive/HRU_project/`, where the next one reads from.

## How we work

Every change goes through a pull request reviewed by another team member — see [CONTRIBUTING.md](CONTRIBUTING.md).
