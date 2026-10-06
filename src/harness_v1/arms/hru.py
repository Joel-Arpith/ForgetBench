"""Arm C: HRU (Hybrid Retention-aware Unlearning) — the project's own method.

Shard-localize (only touch the retain shards least entangled with the forget
authors) -> scrub with NPO -> harden inline: periodically simulate a
relearning attack on a throwaway clone, and if the attack easily lowers the
forget-set loss, replay that exact batch through the main model's NPO loss
with extra weight next step. This targets durability, the thing Arm A pays
full retrain cost for and Arm B doesn't get at all.

ponytail: 'harden inline' here is attack-then-reweight, not a differentiable
bi-level objective (that needs unrolled-SGD autodiff, e.g. via `higher` or
functorch, to backprop through the attack steps themselves). This gets most
of the durability benefit — the model is explicitly trained against batches
that were shown to be relearnable — at plain first-order cost. Upgrade to
`higher`-based MAML-style unrolling if HRU's durability score plateaus and
you need the exact gradient-through-the-attack signal.
"""
import copy
from collections import deque

import torch
import torch.nn.functional as F

from .common import sft, per_example_avg_nll
from .npo_rmu import _Pool


def _shard_localize(forget_ds, retain_ds, n_shards=4):
    """Pick retain shards that share the fewest authors' surface form with the
    forget set (cheap proxy for SISA's 'which shard needs touching' question:
    since forget authors were never in these shards to begin with, none of
    them need retraining — only used to trim the retain replay set)."""
    forget_authors = {qa["author"] for qa in forget_ds}
    shards = [retain_ds[i::n_shards] for i in range(n_shards)]
    return [s for s in shards if not any(qa["author"] in forget_authors for qa in s)]


def _npo_term(model, ref_model, batch, tokenizer, device, beta):
    cur_nll = per_example_avg_nll(model, tokenizer, batch, device)
    with torch.no_grad():
        ref_nll = per_example_avg_nll(ref_model, tokenizer, batch, device)
    logratio = ref_nll - cur_nll
    return (2.0 / beta) * F.softplus(beta * logratio).mean()


class _AttackHardener:
    def __init__(self, forget_pool, tokenizer, attack_steps=3, attack_lr=5e-4,
                 probe_every=5, drop_threshold=0.3, seed=0):
        self.pool = forget_pool
        self.tokenizer = tokenizer
        self.attack_steps = attack_steps
        self.attack_lr = attack_lr
        self.probe_every = probe_every
        self.drop_threshold = drop_threshold
        self.step = 0
        self.hard_batches = deque(maxlen=8)

    def maybe_probe(self, model, device):
        self.step += 1
        if self.step % self.probe_every != 0:
            return
        batch = self.pool.sample(4)
        clone = copy.deepcopy(model)
        opt = torch.optim.SGD(clone.parameters(), lr=self.attack_lr)
        pre = per_example_avg_nll(clone, self.tokenizer, batch, device).mean().item()
        for _ in range(self.attack_steps):
            loss = per_example_avg_nll(clone, self.tokenizer, batch, device).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        post = per_example_avg_nll(clone, self.tokenizer, batch, device).mean().item()
        del clone
        if pre > 0 and (pre - post) / pre > self.drop_threshold:
            self.hard_batches.append(batch)  # attack easily relearned this -> replay w/ extra weight

    def drain_penalty(self, model, ref_model, tokenizer, device, beta, weight=2.0):
        if not self.hard_batches:
            return None
        batch = self.hard_batches.popleft()
        return weight * _npo_term(model, ref_model, batch, tokenizer, device, beta)


def run_hru(model, tokenizer, forget_ds, retain_ds, cfg, n_shards=4):
    localized_retain = [qa for shard in _shard_localize(forget_ds, retain_ds, n_shards) for qa in shard]
    localized_retain = localized_retain or retain_ds  # fallback if localization empties the set

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    pool = _Pool(forget_ds, cfg.seed)
    hardener = _AttackHardener(pool, tokenizer, seed=cfg.seed)

    def extra(model, tokenizer, retain_batch, device):
        forget_batch = pool.sample(len(retain_batch))
        loss = _npo_term(model, ref_model, forget_batch, tokenizer, device, cfg.npo_beta)
        hardener.maybe_probe(model, device)
        hard_penalty = hardener.drain_penalty(model, ref_model, tokenizer, device, cfg.npo_beta)
        if hard_penalty is not None:
            loss = loss + hard_penalty
        return loss

    return sft(model, tokenizer, localized_retain, cfg.unlearn_epochs, cfg.lr, extra_loss_fn=extra)
