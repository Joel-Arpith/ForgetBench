# Evaluation Metrics

| Metric | Definition | Better |
|---|---|---|
| Forget log-prob | Mean log-probability of correct answer tokens on the forget set | Lower |
| Retain log-prob | Same on the retain set (utility) | Higher |
| Truth ratio | P(wrong answers) relative to P(correct answer), TOFU-style | ≈ 1.0 |
| MIA AUC | Min-K% (k = 20%) membership inference, forget vs holdout, answer tokens only | ≈ retrain (0.371) |
| Durability | Forget log-prob **after** a 15-step relearning attack on held-out test probes | Lower |
| Cost | Optimizer updates per deletion request | Lower |

## Relearning attack
15 fine-tune steps, learning rate 5e-5, batch 4, on 40 test paraphrase probes never seen during unlearning ([`eval/attack.py`](../eval/attack.py)).

## Validity notes (self-audit)
- Durability correlates with retain degradation at **r = +0.88**: a whole-answer score partly rewards damage.
- Fix (validation phase): score fact tokens only and report truth ratio after the attack.
- MIA reference is full retraining (0.371), not 0.5, because forget and holdout sets differ slightly.
