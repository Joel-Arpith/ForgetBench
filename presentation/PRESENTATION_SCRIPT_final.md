# ForgetBench · HRU — Final Presentation Script
### 17 slides · 5 speakers · ~15 min + Q&A · matches `ForgetBench_HRU_Final.pptx`

**The one message the panel must leave with:** we built a fair harness, measured twelve methods, found one solid result, audited our own metric, and know exactly which experiment decides whether HRU works.

**Verdict, in one line (everyone memorises this):** *HRU is promising, not yet proven.*

**Speaking order:** Jayanth (1–2) → Kamal (3–4) → Parvez (5–6) → Charan (7–8) → Arpith (9–12) → Charan (13–14) → Jayanth (15–17)

---

## Numbers everyone must know

Lower durability = harder to recover after an attack.

| Method | Durability | Retain (model health) | Cost (updates) |
|---|---|---|---|
| Full retrain | −0.582 | **−0.236** (best) | 150 |
| SISA | −0.457 | −0.459 | 26 |
| Gradient ascent | −1.088 | −1.421 | 40 |
| NPO | −0.762 | −1.594 | 40 |
| RMU | −0.450 | −0.578 | 40 |
| **HRU (unsharded)** | **−1.309** | −1.692 | 120 |
| HRU (sharded) | −0.420 | −0.444 | 120 |

- **Sharding:** scrubbed shard alone −4.18 → inside the 4-shard ensemble −0.50.
- **Audit:** durability correlates with damage, **r = +0.88**.
- **Fair reading:** HRU sits **0.20 below** the damage trend line (more durable than its damage predicts), second only to full retrain (0.22). NPO sits 0.29 above it, the worst.

### Never say
- "HRU is 2.2× better than retraining."
- "The ablation proves hardening works."
- "HRU beats the alternatives."
- Any number that is not on the slides.

---

## JAYANTH — Slides 1–2 (~1 min 45 s)

**Slide 1 — Title.**
Open: **"Can you actually delete something from an AI's memory — and prove it stayed deleted?"**
ForgetBench is our test harness; HRU is our method. "Since the proposal we built the harness, measured twelve methods, and audited our own results. We'll show you what holds up — and what doesn't yet."

**Slide 2 — A Right on Paper, Not in the Model.**
Walk the three boxes: a deletion request is easy for a database, hard for a trained model, because the model learned *patterns*, not a copy of the row. Name the law quickly: GDPR Article 17, India's DPDP Act Section 12, NYT v. OpenAI, the artists' case against Stability AI.
Hand off: **"Legally this is settled. Technically it isn't — Kamal will show where it breaks."**

---

## KAMAL — Slides 3–4 (~2 min)

**Slide 3 — Two Layers, One Word.**
The data layer — deleting records, excluding from training, dropping from retrieval — is solved engineering. The model layer — proving content is gone from the *weights* and stays gone — is not. **"We're not solving data deletion. We're solving weight-level deletion."**

**Slide 4 — And When You Scrub It, It Comes Back.**
Three steps: the scrub passes the test → a light fine-tune → the content comes back. **"It was suppressed, not removed."**
Then the right panel: **"We reproduced this ourselves. NPO drives the forget-set score to −4.87; a 15-step attack brings it back to −0.76."**
Hand off: *"So what matters is whether forgetting survives — and the literature doesn't measure that. Parvez."*

---

## PARVEZ — Slides 5–6 (~1 min 45 s)

**Slide 5 — Four Milestones and One Warning.** One breath each: SISA (exact, retrain-scale), NPO/RMU (cheap, approximate), the relearning-attack warning, TOFU/MUSE (score forgetting once, never after an attack).

**Slide 6 — The Unoccupied Quadrant.** Exact methods are durable but expensive; cheap methods collapse under attack; nothing is cheap *and* durable. "Our design target for HRU is 2–5× a scrub pass. That is a target, not a result — Arpith will show where we actually are."

---

## CHARAN — Slides 7–8 (~1 min 45 s)

**Slide 7 — Objectives.** Four of five done. Say the two honest ones clearly:
- O3: "HRU is built and measured — but our audit found its inline reweight step was not in the pipeline that produced these numbers."
- O4: "Durability is measured for every arm; the audit showed the score partly tracks model damage, so we're repairing it."

**Slide 8 — Hardening Inside the Scrub.** The loop: attack a copy mid-scrub → find what comes back fastest → re-weight the loss toward it. **"Post-hoc hardening can only report a weakness; inline hardening can still fix it."** Read the honest-scope box aloud.
Hand off: *"Arpith will show how it runs and what we measured."*

---

## ARPITH — Slides 9–12 (~4 min) — the results

**Slide 9 — Three Arms, One Harness.** Arm A exact, Arm B approximate, Arm C HRU. **"Same model, probes, seeds and cost accounting for every arm — a difference in results is a difference in method."**

**Slide 10 — The Stack.** Fast. GPT-2 (124M) on one Colab T4 for this phase; 130 fictional profiles so we know exactly what the model could have learned; four axes, durability is the novel one.

**Slide 11 — Progress.** Explain the metric *before* any number: "Durability is the forget-set score after a relearning attack; lower means less recoverable."
- "Unsharded HRU has the lowest score in the study, −1.31, and avoids NPO's over-forgetting — truth ratio 1.29 against 13.17."
- **Say the costs immediately:** "It used 120 updates against NPO's 40, it damaged the model more, and the version we measured didn't include the inline reweight step."
- "Sharded HRU is much worse. That's our clearest finding."

**Slide 12 — Sharding Dilutes Approximate Unlearning.**
**"The scrub worked — the aggregation threw it away."** The scrubbed shard alone reads −4.18; averaged with three untouched shards it reads −0.50, and durability falls from −1.31 to −0.42. SISA averages because, in exact unlearning, the retrained shard has nothing to suppress. Approximate unlearning breaks that assumption.
Hand off: *"Before trusting any of these numbers, we audited them. Charan."*

---

## CHARAN — Slides 13–14 (~2 min 15 s)

**Slide 13 — Self-Audit.**
**"We checked whether our own metric could be fooled — and it partly can."** Across twelve arms, durability correlates with model damage, r = +0.88: the three most durable arms are the three most damaged. The cause: the score averages the whole answer sentence, mostly template words.
Then the fair reading: **"Adjusted for damage, HRU sits 0.20 below the trend line — more durable than its damage predicts — second only to full retraining. NPO is the worst. So HRU's lead isn't only damage, but we can't size the real effect until the metric scores the fact itself."**

**Slide 14 — Verdict: Promising, Not Yet Proven.** Walk the table: ahead of NPO but with 3× its budget; a marginal edge over gradient ascent; full retraining still better on model health; SISA and RMU barely forget. Then the three reasons we can't say "better."
Hand off: *"Jayanth will show the run that settles it."*

---

## JAYANTH — Slides 15–17 (~2 min 15 s) — closes

**Slide 15 — The Experiment That Decides It.** Four steps. The key one: **HRU against SAM-NPO at an equal budget over three seeds — same optimiser, same parameters; the only difference is the reweight step.** About 30–40 minutes of GPU time; the notebook is built. **"We report the result whichever way it lands."**

**Slide 16 — Scope of Claims.** Deliver with confidence: what is built, what is measured, and what we deliberately do not claim.

**Slide 17 — Conclusion.** Three deliverables with honest status. **"The comparison itself is the contribution, whichever way it lands."** Close slowly: **"Turning the right to be forgotten from a legal claim into a measurable engineering property — one that holds up when someone tries to undo it."** Thank the panel.

---

## Q&A — the questions you will get

| Question | Who | Answer |
|---|---|---|
| "So is HRU better or not?" | Charan | "Not proven yet. It's the most durable of the cheap methods on current numbers, but it had more budget, our metric partly rewards damage, and the measured version lacked the key step. Slide 15's run fixes all three." |
| "Why didn't you run the full mechanism?" | Arpith | "Our Colab pipeline used the attack probe only to decide when to stop, not to re-weight the loss. We found that in our own audit. The full mechanism is now implemented and is the next run." |
| "Isn't r = 0.88 fatal to your results?" | Charan | "It's fatal to reading the raw ranking as proof — which is why we don't. It isn't fatal to the method: adjusted for damage HRU is still second only to retraining. The fix is a fact-level metric, already built and tested on all 650 answers." |
| "Why is HRU more expensive than retraining here?" | Parvez | "At our scale retraining is artificially cheap — 400 examples on a small model. In deployment it's closer to 100× a scrub pass, so this setup understates any cost advantage." |
| "Why not just drop sharding?" | Arpith | "That's an option, and unsharded is our best arm. But the cause is the averaging rule, so we're testing weighted and veto aggregation first. If neither works, we drop it and report why." |
| "One seed — how do you know it's not noise?" | Arpith | "We don't, and the slides say so. The deciding run uses three seeds with a paired test." |
| "What's actually novel?" | Charan | "The pieces are published. Training against simulated relearning inside the scrub, and evaluating durability on one fair harness, is ours." |

**If you don't know:** say "We haven't measured that." Never guess a number.
