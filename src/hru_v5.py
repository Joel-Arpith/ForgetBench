"""HRU v5 -- the deciding experiment.

Three fixes over every earlier run, each answering a finding of the audit:

  1. FACT-LEVEL METRIC. The old score averaged the whole answer sentence,
     where only one token is the fact, so suppressing everything looked like
     forgetting (durability vs damage: r = +0.88). `fact_logprob` scores only
     the attribute value ("Calderfen"); `fact_accuracy` checks whether greedy
     generation still names it. Both are measured before AND after the attack.

  2. THE REAL MECHANISM. Ported from the July `arms/hru.py` `_AttackHardener`
     and upgraded to per-example weights: every `probe_every` steps, attack a
     throwaway clone with a few SGD steps on a forget batch, measure each
     example's relative loss drop, and up-weight the examples the attack
     relearned fastest in the following NPO steps. Earlier pipelines only
     used the probe to decide when to stop, so the mechanism was never tested.

  3. EQUAL BUDGET. Every arm gets the same number of optimizer updates.
     HRU's probe steps run on clones and never update the model; they are
     reported separately as overhead, not hidden.

The deciding comparison is HRU-v5 vs SAM-NPO: same optimiser, same params,
same budget, same seeds. The only difference is the reweighting.
"""
import copy
import random
import time

import torch
import torch.nn.functional as F

_MARKERS = (" born in", " works as", " hobby is", " color is")


# ------------------------------------------------------------- fact spans
def split_fact(answer):
    """'X was born in Calderfen.' -> ('X was born in', ' Calderfen').

    The value keeps its leading space so GPT-2's BPE splits cleanly at the
    boundary. Falls back to the last word if no template marker is found.
    """
    a = answer.rstrip(".")
    for m in _MARKERS:
        i = a.rfind(m)
        if i >= 0:
            j = i + len(m)
            return a[:j], a[j:]
    k = a.rfind(" ")
    return a[:k], a[k:]


@torch.no_grad()
def fact_logprob(U, model, tokenizer, qa_pairs):
    """Mean per-token log-prob of the FACT tokens only. Lower = forgotten."""
    model.eval()
    vals = []
    for q, a in qa_pairs:
        before, val = split_fact(a)
        pre = f"Q: {q}\nA: {before}"
        p_ids = tokenizer(pre, return_tensors="pt").to(U.DEVICE)["input_ids"]
        f_ids = tokenizer(pre + val, return_tensors="pt").to(U.DEVICE)["input_ids"]
        n = f_ids.shape[1] - p_ids.shape[1]
        if n <= 0:
            continue
        logits = model(f_ids).logits[:, :-1, :]
        lp = F.log_softmax(logits, -1).gather(-1, f_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        vals.append(lp[0, -n:].mean().item())
    return sum(vals) / max(len(vals), 1)


@torch.no_grad()
def fact_accuracy(U, model, tokenizer, qa_pairs, limit=50):
    """Share of questions where greedy generation still names the fact."""
    hits, n = 0, 0
    for q, a in qa_pairs[:limit]:
        val = split_fact(a)[1].strip().lower()
        out = U.generate_greedy(model, tokenizer, q).lower()
        hits += int(val in out)
        n += 1
    return hits / max(n, 1)


# ------------------------------------------------------------- per-example NPO
def _seq_logprob(U, model, tokenizer, batch, grad):
    enc, labels = U.build_batch(tokenizer, batch, answer_only=True)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1, :]
    tgt = labels[:, 1:]
    mask = (tgt != -100).float()
    lp = F.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def npo_per_example(U, policy, ref, tokenizer, batch, beta):
    diff = _seq_logprob(U, policy, tokenizer, batch, True) - _seq_logprob(U, ref, tokenizer, batch, False)
    return -F.logsigmoid(-beta * diff) * (2.0 / beta)


# ------------------------------------------------------------- the hardener
class AttackHardener:
    """Every `probe_every` steps: SGD-attack a clone on a forget batch for
    `attack_steps` steps, then weight each example by how fast it came back.
    Only clones are attacked; the model being unlearned is never touched here.
    """

    def __init__(self, U, forget_qa, tokenizer, probe_every=5, attack_steps=3,
                 attack_lr=5e-4, batch_size=8, alpha=4.0, w_max=3.0):
        self.U, self.pool, self.tok = U, list(forget_qa), tokenizer
        self.probe_every, self.attack_steps, self.attack_lr = probe_every, attack_steps, attack_lr
        self.batch_size, self.alpha, self.w_max = batch_size, alpha, w_max
        self.hard, self.step, self.probe_updates = [], 0, 0

    def maybe_probe(self, model):
        self.step += 1
        if self.step % self.probe_every:
            return
        U = self.U
        batch = random.sample(self.pool, min(self.batch_size, len(self.pool)))
        clone = copy.deepcopy(model)
        for p in clone.parameters():
            p.requires_grad_(True)
        opt = torch.optim.SGD(clone.parameters(), lr=self.attack_lr)
        clone.train()
        with torch.no_grad():
            pre = -_seq_logprob(U, clone, self.tok, batch, False)          # per-example NLL
        for _ in range(self.attack_steps):
            loss = U.lm_loss(clone, self.tok, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            self.probe_updates += 1
        with torch.no_grad():
            post = -_seq_logprob(U, clone, self.tok, batch, False)
        del clone
        drop = ((pre - post) / pre.clamp(min=1e-6)).clamp(min=0)          # relative relearning speed
        w = (1.0 + self.alpha * drop).clamp(max=self.w_max)
        self.hard = [(ex, float(wi)) for ex, wi in zip(batch, w) if wi > 1.05]

    def weighted_term(self, policy, ref, beta):
        if not self.hard:
            return None
        batch = [ex for ex, _ in self.hard]
        w = torch.tensor([wi for _, wi in self.hard], device=self.U.DEVICE)
        per = npo_per_example(self.U, policy, ref, self.tok, batch, beta)
        return (per * w).sum() / w.sum()


# ------------------------------------------------------------- one scrub routine for both SAM arms
def sam_scrub(U, policy, ref, tokenizer, forget_qa, retain_qa, updates=120, lr=5e-5,
              beta=0.1, rho=0.05, retain_weight=1.0, batch_size=4,
              layer_frac=(0.25, 0.75), hardener=None, cost=None, desc="sam"):
    """SAM-optimised NPO on mid-stack MLP weights. With `hardener` = HRU-v5;
    without = the SAM-NPO control. Identical in every other respect."""
    layers = U.get_layers(policy)
    n = len(layers); lo, hi = int(n * layer_frac[0]), max(int(n * layer_frac[1]), int(n * layer_frac[0]) + 1)
    for p in policy.parameters():
        p.requires_grad_(False)
    live = [p for i in range(lo, min(hi, n)) for name, p in layers[i].named_parameters() if "mlp" in name]
    for p in live:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(live, lr=lr)
    policy.train(); t0 = time.time()

    for s in range(updates):
        f = random.sample(forget_qa, min(len(forget_qa), batch_size))
        r = random.sample(retain_qa, min(len(retain_qa), batch_size))

        def objective():
            loss = npo_per_example(U, policy, ref, tokenizer, f, beta).mean() \
                   + retain_weight * U.lm_loss(policy, tokenizer, r)
            if hardener is not None:
                extra = hardener.weighted_term(policy, ref, beta)
                if extra is not None:
                    loss = loss + extra
            return loss

        loss = objective(); opt.zero_grad(); loss.backward()
        with torch.no_grad():
            grads = [p.grad for p in live if p.grad is not None]
            gn = torch.norm(torch.stack([g.norm() for g in grads])) if grads else torch.tensor(0.0)
            eps = []
            for p in live:
                e = None if p.grad is None else p.grad * (rho / (gn + 1e-12))
                if e is not None:
                    p.add_(e)
                eps.append(e)
        loss2 = objective(); opt.zero_grad(); loss2.backward()
        with torch.no_grad():
            for p, e in zip(live, eps):
                if e is not None:
                    p.sub_(e)
        opt.step()

        if hardener is not None:
            for p in policy.parameters():
                p.requires_grad_(True)       # clone must be fully trainable for a fair attack
            hardener.maybe_probe(policy)
            for p in policy.parameters():
                p.requires_grad_(False)
            for p in live:
                p.requires_grad_(True)

        if cost is not None:
            cost["updates"] += 1
        if s % 20 == 0:
            print(f"  [{desc}] update {s} loss {loss.item():.4f}")

    for p in policy.parameters():
        p.requires_grad_(True)
    if cost is not None:
        cost["seconds"] += time.time() - t0
        if hardener is not None:
            cost["probe_updates"] += hardener.probe_updates
    return policy


# ------------------------------------------------------------- evaluation
def evaluate(U, model, tokenizer, data, attack_steps=15, seed=42, gen_limit=50):
    """Every metric before and after the relearning attack (test probes)."""
    out = {
        "fact_lp": fact_logprob(U, model, tokenizer, data["forget_qa"]),
        "fact_acc": fact_accuracy(U, model, tokenizer, data["forget_qa"], gen_limit),
        "truth_ratio": U.truth_ratio(model, tokenizer, data["forget_qa"], data["forget_perturbed"]),
        "retain_fact_lp": fact_logprob(U, model, tokenizer, data["retain_qa"]),
        "retain_acc": fact_accuracy(U, model, tokenizer, data["retain_qa"], gen_limit),
        "general_ppl": U.general_perplexity(model, tokenizer, data["general_text"]),
    }
    U.set_seed(seed)
    atk = copy.deepcopy(model)
    for p in atk.parameters():
        p.requires_grad_(True)
    U.relearn_attack(atk, tokenizer, data["attack_test_qa"], steps=attack_steps)
    out["fact_lp_after"] = fact_logprob(U, atk, tokenizer, data["forget_qa"])
    out["fact_acc_after"] = fact_accuracy(U, atk, tokenizer, data["forget_qa"], gen_limit)
    out["truth_ratio_after"] = U.truth_ratio(atk, tokenizer, data["forget_qa"], data["forget_perturbed"])
    out["retain_fact_lp_after"] = fact_logprob(U, atk, tokenizer, data["retain_qa"])
    del atk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
