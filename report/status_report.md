# HRU — Status Report and Technical Assessment

**Project:** ForgetBench · HRU (Hybrid Retention-aware Unlearning)
**Team:**

| Name | Register no. | Programme | Email |
|---|---|---|---|
| Bulla Joel Arpith | 25WU0103011 | | |
| Kanchusthambham Jayanth | 25WU0103021 | B.Tech, BIC | jayanth.k_2029@woxsen.edu.in |
| Macha Sai Kamal | 25WU0103026 | B.Tech CSE, BIC | saikamal.m_2029@woxsen.edu.in |
| Shaik Shahul Parvez | 25WU0103036 | B.Tech, BIC | shahul.shaik_2029@woxsen.edu.in |
| Damera Sai Sri Ranga Charan | 25WU0102048 | B.Tech CSE (AI & ML) | charan.damera_2029@woxsen.edu.in |

**Mentor:** Prof. Geetha Tripathi, Woxsen University
**Date:** 1 October 2026
**Status:** Harness complete and producing results. Core HRU mechanism not yet tested. Evaluation metric needs repair before any durability claim is final.

---

## 1. Executive summary

**The plain verdict.** The harness works and has produced a full set of measurements across twelve method variants. One finding from those measurements is solid and worth reporting. The headline durability result, HRU at −1.309 against full retraining at −0.582, is **not yet evidence that HRU works**, for three reasons that this report documents:

1. **HRU's defining mechanism was never in the pipeline that produced the numbers.** The attack-then-reweight loop described in the proposal exists in the July codebase (`arms/hru.py`, `_AttackHardener`). The Colab notebooks that generated every measurement replaced it with a probe that only decided *when to stop*, and never changed what the model learned. Nothing measured so far tests HRU's novel idea.
2. **The durability metric mostly measures how much the model was damaged.** Across all twelve arms, durability correlates with retain-set degradation at **r = +0.88**. The three "most durable" arms are exactly the three most damaged models. Naive gradient ascent, included as the baseline expected to fail, scores 83% of HRU's durability at one third of the cost.
3. **The winning arm differs from the NPO baseline in four ways at once**: three times the scrub budget, sharpness-aware optimisation, a restricted parameter set, and no sharding. The gain cannot be attributed to any one of them.

**What is solid.** SISA-style sharding with logit averaging *dilutes approximate unlearning*. A scrubbed shard alone reaches a forget log-probability of −4.179. Averaged with three untouched shards, the ensemble reads −0.502, and durability collapses from −1.309 (unsharded) to −0.420 (sharded). SISA assumes the deleted shard has nothing to suppress, which holds for exact unlearning and fails for approximate unlearning. This is a mechanism with a measured effect and a clear explanation.

**What to do next** (Section 9) is concrete and cheap. Repair the metric to score fact tokens only. Port the real HRU mechanism into the pipeline. Run an equal-budget comparison over three seeds. On a T4 that is roughly half an hour of GPU time, and it is the experiment that can actually establish whether HRU works.

---

## 2. Corrections to claims made earlier

Several statements in the progress slide (slide 11), its speaker notes, and presentation script v2 are not supported by the data. They were written by Claude during the analysis and are corrected here.

| Earlier claim | Status | Supportable replacement |
|---|---|---|
| "The ablation proves the gain comes from the hardening loop, not from scrubbing harder" (scrub-only −0.459 vs HRU −1.309) | **Wrong.** Those two arms differ in sharding *and* hardening. The correct pairing holds sharding fixed: scrub-only −0.459 vs sharded HRU −0.420, so hardening had no effect. It could not have had one, because the probe never fed back into the loss. | "The clear effect is that logit-averaged sharding dilutes approximate unlearning. The inline hardening mechanism has not been tested yet." |
| "HRU is 2.2× more durable than full retraining" | **Not supportable.** The metric tracks model damage (r = +0.88). A ratio of log-probabilities is also not a meaningful "times more" quantity. | "Unsharded HRU shows the lowest post-attack forget likelihood in the study (−1.309 vs −0.582). We are validating whether that is fact-specific forgetting or broad suppression." |
| "HRU has the healthiest MIA, 0.433, closest to the ideal 0.5" | **Wrong baseline.** Full retraining never saw the forget set *or* the holdout set, yet scores 0.371. The two sets differ in distribution, so 0.371 is the effective "never trained on it" reference, not 0.5. | Report MIA relative to retrain's 0.371. On that reading SISA (0.381) and RMU (0.412) sit closer than HRU. |
| Slide 11 row label "HRU inline-hardened" | **Mislabelled.** That arm contains no working inline hardening. | "HRU (unsharded, SAM scrub)" |

---

## 3. What has been built

**Pipeline.** Five Colab notebooks, all persisting to `MyDrive/HRU_project` on Google Drive, plus two follow-ups:

| Notebook | Purpose | Status |
|---|---|---|
| 01 Data + base model | Synthetic TOFU-style corpus (130 fictional profiles: 20 forget, 80 retain, 30 holdout never trained on), GPT-2 fine-tune, sanity asserts | Run, passed |
| 02 Harness + reference | Full metric suite on the un-unlearned model | Run |
| 03 Arms A & B | Full retrain, SISA, sharded control, gradient ascent, NPO, RMU | Run |
| 04 Arm C (v3) | HRU with SAM scrub, config search, ablation, shard-count sweep | Run |
| 05 Benchmark | Recovery curves, cost vs durability, amortised cost, sequential deletions | Run |
| 04b Aggregation sweep | Weighted vs veto shard aggregation, no training | **Built, not run** |
| 06 HRU v4 | Cost-optimised variant (no shard, plateau stop, alternating SAM) | **Built, not run** |

**Two codebases exist and have diverged.** The July harness in this folder (`harness.py`, `arms/`, `eval/`) has never produced results; its `results/` directory is empty. The Colab notebooks produced every number in this report but do not contain the July `_AttackHardener`. These should be consolidated (Section 9, step 2).

**Evaluation suite (per arm):**

| Metric | What it measures | Direction |
|---|---|---|
| `forget_logprob` | Mean log-probability of gold answer tokens on the forget set | lower = more forgotten |
| `retain_logprob` | Same, on the retain set | higher = utility kept |
| `truth_ratio` | P(wrong answers) / P(correct answer), TOFU-style | ~1.0 = no preference, like a model that never saw the data |
| `mia_auc` | Min-K% membership inference, forget vs holdout | compare to retrain's 0.371 (see §2) |
| `durability` | `forget_logprob` after a 15-step relearning attack on held-out test paraphrases | lower = less recoverable |
| `cost` | Optimizer updates per deletion request | lower = cheaper |

**Methodological safeguards already in place:** dev/test split of attack probes (hardening may only see dev phrasings), a sharded control arm that isolates the sharding effect, answer-only loss masking, global seeding, and comma-safe CSV logging.

---

## 4. Method: HRU as designed vs as measured

### 4.1 As designed (proposal, slide 8, `arms/hru.py`)

1. **Shard-localize.** Restrict the retain replay set to shards least entangled with the forget authors.
2. **Scrub.** NPO loss on the forget set against a frozen reference.
3. **Harden inline.** Every 5 steps, clone the model and run 3 SGD attack steps on a forget batch. If the attack lowers that batch's loss by more than 30%, store the batch as "hard". On the next step, replay it through the NPO loss at 2× weight.

Step 3 is the contribution. It shapes the scrub around its own failure mode while training budget remains.

### 4.2 As measured (Colab v2/v3)

1. **Shard** via SISA-style data split; ensemble by averaging shard logits.
2. **Scrub** with NPO, optimised with Sharpness-Aware Minimization (SAM) on mid-stack MLP parameters only (22.8% of weights).
3. **"Harden"**: after each scrub round, run a relearning attack on a deep copy and record leakage. **Leakage was used only as an early-stop test.** It never re-weighted the loss. The stop threshold (−3.0, later −1.1) was never reached, so every configuration ran all six rounds.

When early stopping never fires, the measured "hardening loop" is mathematically a plain SAM scrub. The only difference is that the probe resets the random seed, which reshuffles batch order. That is why sharded HRU (−0.420) and scrub-only (−0.459) are statistically indistinguishable.

### 4.3 v4 (built, not run)

v4 removes the shard stage, deletes the probes, stops on a sharpness plateau, and applies SAM on alternating steps. It cuts the projected cost from 240 to 40–120 updates. **But v4 optimises a method whose value is not yet established, and it contains no attack-then-reweight mechanism.** What remains is essentially SAM-based NPO on a parameter subset. That is close to the approach of arXiv:2502.05374, so it carries little novelty on its own. Its cost techniques should be applied *after* the mechanism is validated, not instead of validating it.

---

## 5. Experimental setup

| Item | Value |
|---|---|
| Base model | GPT-2 (124M), fine-tuned 6 epochs on forget + retain |
| Forget set | 20 fictional profiles × 5 facts = 100 QA pairs |
| Retain set | 80 profiles = 400 QA pairs |
| Holdout (MIA non-members) | 30 profiles = 150 QA pairs, never trained on |
| Attack | 15 fine-tune steps (lr 5e-5) on paraphrased probes of the forget facts; 40 dev / 40 test probes |
| Hardware | Google Colab, single T4 GPU |
| Seeds | **1** (FAST_MODE). Rows marked seeds=2 mix two separate runs (§7.4) |
| Cost unit | Optimizer updates. One NPO scrub pass = 40 updates = "1×" |

---

## 6. Results

### 6.1 Main table (single seed, notebook 05)

| Arm | Forget LP | Retain LP | Truth ratio | MIA | **Durability** | Updates |
|---|---|---|---|---|---|---|
| A_retrain | −0.259 | **−0.236** | **1.007** | 0.371 | −0.582 | 150 |
| A_shard_control | −0.388 | −0.390 | 0.994 | 0.391 | −0.380 | 0 |
| A_sisa | −0.478 | −0.459 | 1.023 | 0.381 | −0.457 | 26 |
| B_ga | −4.100 | −1.421 | 1.665 | 0.314 | −1.088 | 40 |
| B_npo | −4.871 | −1.594 | 13.166 | 0.275 | −0.762 | 40 |
| B_rmu | −0.567 | −0.578 | 0.986 | 0.412 | −0.450 | 40 |
| C_scrub_only (sharded) | −0.604 | −0.484 | 1.155 | 0.363 | −0.459 | 120 |
| C_hru_full (sharded) | −0.502 | −0.444 | 1.059 | 0.358 | −0.420 | 120 |
| **C_harden_noshard** | −4.179 | −1.692 | 1.292 | 0.433 | **−1.309** | 120 |
| C_hru_k2 | −0.701 | −0.415 | 1.096 | 0.416 | −0.378 | 120 |
| C_hru_k4 | −0.532 | −0.460 | 1.088 | 0.362 | −0.439 | 120 |
| C_hru_k8 | −0.702 | −0.694 | 1.029 | 0.361 | −0.670 | 120 |

HRU updates are reported as SAM optimizer updates (120). Notebook output shows 360 "steps" because SAM uses two forward/backward passes per update and the diagnostic probes added another 120. Neither of those changes what the model learned, so they are excluded here for comparability with notebook 03's single-pass baselines.

### 6.2 Cost

Normalised to one NPO scrub pass (40 updates = 1×): SISA 0.65×, full retrain 3.75×, every HRU variant 3.0× (6.0× including probe overhead). The design target was 2–5×. At this toy scale full retraining is artificially cheap (400 examples). In deployment it is closer to ~100× a scrub pass, so this harness *understates* any cost advantage over retraining.

---

## 7. Findings

### 7.1 Sharding with logit averaging dilutes approximate unlearning — **solid**

| | Forget LP | Durability |
|---|---|---|
| Scrubbed shard 0, alone | −4.179 | — |
| Same shard inside 4-shard ensemble | −0.502 | −0.420 |
| Same procedure, no sharding | −4.179 | −1.309 |

Three untouched shards, which never saw the forget set, sit near baseline and outvote the one suppressing shard 3-to-1. SISA's averaging is correct for *exact* unlearning, where the retrained shard has nothing to suppress. It is wrong for *approximate* unlearning, where the scrubbed shard carries a suppression signal the others cannot express. Notebook 04b tests whether weighted or veto (elementwise-minimum) aggregation recovers it.

*Caveat:* the unsharded row averages two runs (§7.4), so the size of the effect is approximate. Its direction, on a 3× gap, is not in doubt.

### 7.2 The durability metric is confounded with model damage — **critical**

Correlation between retain degradation and durability across all twelve arms: **r = +0.88.**

| Rank by durability | Arm | Durability | Retain |
|---|---|---|---|
| 1 | C_harden_noshard | −1.309 | −1.692 |
| 2 | B_ga | −1.088 | −1.421 |
| 3 | B_npo | −0.762 | −1.594 |
| 4 | C_hru_k8 | −0.670 | −0.694 |
| 5 | A_retrain | −0.582 | **−0.236** |

The top three are the three most damaged models. Three further pieces of evidence point at the cause:

- **The metric is dominated by template tokens.** `forget_logprob` averages over the whole answer ("Soren Solis was born in Calderfen."), and most of those tokens are the template or the name copied from the question. Only one token is the fact. The base model's forget-set score (−0.210) barely exceeds never-seen holdout profiles (−0.271), a gap of 0.06 nats, even though it *generates* the correct facts.
- **Retraining looks like it "knows" the forget set.** Full retrain never saw a single forget fact, yet scores −0.259, higher than the sharded control (−0.388), which did see them. Retrain simply learned the answer template better.
- **The attack lowers the score of models that never unlearned anything.** The reference and retrain recovery curves fall under attack (reference roughly −0.21 → −0.57 over 40 steps). A genuine relearning attack should raise forget likelihood. These curves fall because the attack data uses a different answer template ("X grew up in …"), which pulls probability away from the original phrasing.

**The encouraging part.** Measured against a least-squares fit of durability on retain damage, unsharded HRU is more durable than its damage predicts by 0.20, second only to full retraining (0.22). NPO is the worst outlier, 0.29 *less* durable than predicted. HRU's lead is not only damage, but the size of any real effect is unknown until the metric is fixed.

**Implication.** The current durability number rewards suppressing the whole answer string, which also damages the model. It does not isolate whether the *fact* comes back. Every durability comparison in §6 should be treated as preliminary until the metric is repaired (§9, step 1).

### 7.3 Truth ratio — the one encouraging fact-specific signal

Truth ratio compares the correct answer against wrong answers that differ only in the fact token, so it is not template-dominated. A model that has truly forgotten should be indifferent (~1.0), which is exactly what full retraining shows (1.007).

| Arm | Truth ratio |
|---|---|
| A_retrain (ideal) | 1.007 |
| C_harden_noshard | 1.292 |
| B_ga | 1.665 |
| B_npo | 13.166 |

Among the three aggressive scrubbers, HRU (unsharded) is the closest to retrain-like indifference. NPO at 13.2 has overshot into actively preferring wrong answers, a recognisable over-forgetting signature. This is pre-attack only. Measuring truth ratio *after* the attack is the single most informative addition available.

### 7.4 Data hygiene

- **Single seed.** No result in this report carries statistical significance.
- **Contaminated rows.** C_harden_noshard, C_hru_full and C_scrub_only show seeds=2. They average one run from the v2 pipeline (plain NPO) with one from v3 (SAM). Those are different algorithms. `results.csv` should be wiped and regenerated before any number is quoted externally.
- **MIA baseline** is 0.371, not 0.5 (see §2).

### 7.5 Sequential deletions

Across three successive deletion requests, retain log-probability moved −1.95 → −2.22 → −1.82 and general perplexity 205 → 193 → 225. Three noisy points do not support a trend either way. The reference model's perplexity is not in these results, so the level of fluency damage needs that baseline before it can be read.

---

## 8. Limitations

1. The core HRU mechanism is untested (§4).
2. The durability metric is confounded with model damage (§7.2).
3. Single seed; some rows mix pipeline versions (§7.4).
4. Toy scale: GPT-2 124M, 100 forget facts. The template confound is partly an artefact of short templated answers, and would weaken on free-form data such as TOFU's real profiles or MUSE.
5. The forget set and holdout set differ in distribution, which biases MIA (§2).
6. Prior-art check incomplete. SAM for relearning-resilient unlearning (arXiv:2502.05374) and meta-learned tamper resistance (TAR, Meta-Unlearning) are close to HRU's territory. Novelty must be argued specifically: first-order replay of attack-identified hard batches inside the scrub.

---

## 9. Plan: what would make HRU a defensible result

In priority order. Steps 1–4 together take roughly 30–40 minutes of T4 time.

### Step 1 — Repair the metric (≈1 hour of code, no GPU)
- Score **fact-span tokens only**: the log-probability of the attribute value ("Calderfen"), not the whole sentence.
- Add **post-attack truth ratio** and **post-attack retain log-prob** to every durability measurement.
- Add **fact accuracy under greedy generation**, before and after attack. It is the most interpretable number for a panel: "the model named the deleted city in X% of cases after the attack."
- Report durability against retain as a **frontier plot**, not a ranked list. A method only "wins" if it sits below the curve the other methods trace.

### Step 2 — Port the real HRU mechanism (v5)
Bring `_AttackHardener` from `arms/hru.py` into the Colab pipeline, upgraded from batch replay to per-example weighting. Use each example's loss drop on the attacked clone as its weight in the next NPO step. Keep SAM and the selective parameters as optional components.

### Step 3 — Equal-budget comparison (3 seeds, no sharding, 120 updates each)

| Arm | Isolates |
|---|---|
| NPO-120 | baseline at matched budget |
| GA-120 | damage-only control |
| SAM-NPO-120 | current "HRU" without the mechanism |
| **HRU-v5-120** | SAM-NPO **+ attack-then-reweight** |

**HRU-v5 vs SAM-NPO, at equal budget, on post-attack fact recovery** is the comparison that proves or refutes the proposal's central claim. Report it with a paired test across seeds.

### Step 4 — Run notebook 04b
It needs no training. It answers whether veto aggregation rescues sharding, which would turn §7.1 into either a fix or a fully diagnosed negative result.

### Step 5 — Then optimise cost
Apply v4's techniques to v5: no shard stage (unless 04b rescues it), plateau stopping, alternating SAM, selective parameters. The July hardener is already light (3 SGD steps on 4 examples every 5 steps, roughly 60% overhead), so v5 should land near 2× a scrub pass, inside the 2–5× target.

### Step 6 — Consolidate
One codebase. Either port the notebook pipeline back into `harness.py`/`arms/`, or archive the July harness, so the method described in the report is the method that produced its numbers.

---

## 10. How to present this honestly

A mid-project review rewards a team that can say exactly what it knows. A defensible version of the update:

> "The harness is complete and running all twelve variants. Our clearest finding is that SISA-style sharding dilutes approximate unlearning: a scrubbed shard that alone reaches −4.18 drops to −0.50 inside the ensemble, because three untouched shards outvote it. While auditing the results we found our durability metric tracks overall model damage (r = 0.88), so we are repairing it to score fact tokens directly. We have also confirmed that our inline hardening mechanism, the core of HRU, was not in the pipeline that produced these numbers. The next run tests it head-to-head against an equal-budget baseline. We expect to answer whether HRU works within the next experiment cycle."

That turns an audit into a demonstration of rigour. It is also the version that survives a sharp question.

---

## Appendix A — Cost accounting

| Quantity | Counts | Use |
|---|---|---|
| `updates` | optimizer steps that change the model | cross-method comparison |
| `steps` | forward/backward passes (SAM = 2 per update) | FLOP proxy |
| `probe_updates` | attack steps on throwaway clones | method overhead (HRU-v5) or validation (v4) |
| `seconds` | wall-clock | sanity check; varies with Colab load |

## Appendix B — Files

| Artifact | Location |
|---|---|
| July harness (contains `_AttackHardener`) | `03_PROJECTS/active/hru-unlearning/arms/hru.py` |
| Colab notebooks, data, checkpoints, `results.csv` | Google Drive `MyDrive/HRU_project/` |
| Shared utils / v3 / v4 modules | written by notebooks 01, 04 v3, 06 into `MyDrive/HRU_project/` |
| Progress deck | `~/Downloads/Machine_Unlearning_UPDATED.pptx` (slide 11 needs the §2 corrections) |

## Appendix C — References

- Bourtoule et al., *Machine Unlearning* (SISA), IEEE S&P 2021.
- Guo et al., *Certified Data Removal from Machine Learning Models*, ICML 2020, arXiv:1911.03030.
- Zhang et al., *Negative Preference Optimization*, arXiv:2404.05868.
- Li et al., *The WMDP Benchmark* (RMU), arXiv:2403.03218.
- Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*, arXiv:2401.06121.
- Shi et al., *MUSE: Machine Unlearning Six-Way Evaluation*, arXiv:2407.06460.
- *Jogging the Memory of Unlearned Models Through Targeted Relearning Attacks*, arXiv:2406.13356.
- *Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs*, arXiv:2505.16831.
- *Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond*, arXiv:2502.05374.
- *Improving LLM Unlearning Robustness via Random Perturbations*, arXiv:2501.19202.
- Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021.
- Shi et al., *Detecting Pretraining Data from Large Language Models* (Min-K% Prob), arXiv:2310.16789.
- Chen et al., *When Machine Unlearning Jeopardizes Privacy*, CCS 2021.
- Duan et al., 2024, on the weakness of membership inference against LLMs, as discussed in arXiv:2412.13475.
