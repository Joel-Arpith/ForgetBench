from arms.common import avg_answer_nll


def forget_retain_quality(model, tokenizer, forget_ds, retain_ds):
    """TOFU/MUSE-style pair: forget NLL should rise (model no longer confident
    about forget facts), retain NLL should stay close to its pre-unlearn value
    (model still knows everything else)."""
    return {
        "forget_nll": avg_answer_nll(model, tokenizer, forget_ds),
        "retain_nll": avg_answer_nll(model, tokenizer, retain_ds),
    }
