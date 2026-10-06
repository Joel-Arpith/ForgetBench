import copy

from arms.common import avg_answer_nll, sft


def run_attack(model, tokenizer, forget_ds, retain_ds, cfg, pristine_forget_nll):
    """Simulated relearning attack: briefly fine-tune the unlearned model on
    retain-adjacent data (never the forget set itself) and see how much
    forget-set confidence resurfaces. This is the durability check that
    distinguishes 'scrubbed' (Arm B) from 'actually gone' (Arm A) — and is
    the metric HRU (Arm C) is designed to win on.
    """
    nll_before_attack = avg_answer_nll(model, tokenizer, forget_ds)

    clone = copy.deepcopy(model)
    attack_batch = retain_ds[: max(8, len(retain_ds) // 10)]
    clone = sft(clone, tokenizer, attack_batch, epochs=1, lr=cfg.lr,
                batch_size=min(4, max(1, len(attack_batch))))
    # a handful of optimizer steps, not full epochs, is the point of "attack_steps";
    # sft() runs epoch-granular, so a small slice of retain_ds approximates cfg.attack_steps
    nll_after_attack = avg_answer_nll(clone, tokenizer, forget_ds)

    resurfacing = max(0.0, nll_before_attack - nll_after_attack)
    headroom = max(nll_before_attack - pristine_forget_nll, 1e-6)
    durability = 1.0 - min(1.0, resurfacing / headroom)

    return {
        "forget_nll_pre_attack": nll_before_attack,
        "forget_nll_post_attack": nll_after_attack,
        "resurfacing": resurfacing,
        "durability": durability,
    }
