"""Arm B: approximate unlearning. Scrubs a model that already knows the
forget set (as opposed to Arm A, which never lets the model see it)."""
import copy
import torch
import torch.nn.functional as F

from .common import add_lora, sft, per_example_avg_nll, find_decoder_layers, freeze_all_but_layers


def _npo_extra_loss(ref_model, forget_pool, beta):
    """Returns an extra_loss_fn(model, tokenizer, batch_qas, device) for sft().
    Each retain batch also gets one forget mini-batch mixed in, penalized by
    NPO: push p_theta(forget) below p_ref(forget) without collapsing to zero
    (unlike a naive gradient-ascent-on-forget-loss, which destabilizes fast)."""
    def extra(model, tokenizer, retain_batch, device):
        forget_batch = forget_pool.sample(len(retain_batch))
        cur_nll = per_example_avg_nll(model, tokenizer, forget_batch, device)
        with torch.no_grad():
            ref_nll = per_example_avg_nll(ref_model, tokenizer, forget_batch, device)
        logratio = ref_nll - cur_nll  # positive if model is now MORE confident than ref
        return (2.0 / beta) * F.softplus(beta * logratio).mean()
    return extra


class _Pool:
    """ponytail: plain random sampling with replacement, not epoch-balanced —
    fine at forget-set sizes in the hundreds; swap to a proper sampler if the
    forget set grows past what one shuffle-per-epoch would cover."""
    def __init__(self, items, rng_seed=0):
        import random
        self.items = items
        self.rng = random.Random(rng_seed)

    def sample(self, k):
        return self.rng.choices(self.items, k=k)


def run_npo(model, tokenizer, forget_ds, retain_ds, cfg):
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    pool = _Pool(forget_ds, cfg.seed)
    extra = _npo_extra_loss(ref_model, pool, cfg.npo_beta)
    return sft(model, tokenizer, retain_ds, cfg.unlearn_epochs, cfg.lr, extra_loss_fn=extra)


def _hook_capture(layer, key, store):
    def fn(_, __, output):
        store[key] = output[0] if isinstance(output, tuple) else output
    return layer.register_forward_hook(fn)


def _calibrate_target(ref_model, tokenizer, forget_pool, layer_idx, device, multiplier, rng_seed):
    """RMU's control vector needs to live at the SAME scale as real activations
    at that layer, or the MSE loss is either negligible or blows up the model —
    so measure the forget-set's actual activation norm at the target layer
    first, then set target magnitude = multiplier * that norm (per the paper,
    not a hand-picked constant)."""
    torch.manual_seed(rng_seed)
    direction = torch.randn(ref_model.config.hidden_size)
    direction = direction / direction.norm()

    layers_ref = find_decoder_layers(ref_model)
    captured = {}
    handle = _hook_capture(layers_ref[layer_idx], "act", captured)
    with torch.no_grad():
        probe = forget_pool.sample(8)
        per_example_avg_nll(ref_model, tokenizer, probe, device)
    handle.remove()
    avg_norm = captured["act"].norm(dim=-1).mean().item()
    return direction.to(device) * avg_norm * multiplier


def _rmu_extra_loss(ref_model, forget_pool, layer_idx, retain_weight, target, device):
    """Push forget-set hidden states at one layer toward the calibrated random
    direction (misdirection); anchor retain-set hidden states at that same
    layer to the frozen reference (capability preservation). Classic RMU."""
    layers_ref = find_decoder_layers(ref_model)

    def extra(model, tokenizer, retain_batch, device):
        forget_batch = forget_pool.sample(len(retain_batch))
        layers_cur = find_decoder_layers(model)
        captured = {}
        h_cur = _hook_capture(layers_cur[layer_idx], "cur", captured)
        h_ref = _hook_capture(layers_ref[layer_idx], "ref", captured)
        try:
            per_example_avg_nll(model, tokenizer, forget_batch, device)  # populate captured['cur']
            forget_act = captured["cur"]
            forget_loss = F.mse_loss(forget_act, target.expand_as(forget_act))

            per_example_avg_nll(model, tokenizer, retain_batch, device)
            retain_cur = captured["cur"]
            with torch.no_grad():
                per_example_avg_nll(ref_model, tokenizer, retain_batch, device)
                retain_ref = captured["ref"]
            retain_loss = F.mse_loss(retain_cur, retain_ref)
        finally:
            h_cur.remove(); h_ref.remove()
        return forget_loss + retain_weight * retain_loss

    return extra


def run_rmu(model, tokenizer, forget_ds, retain_ds, cfg, layer_idx=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    n_layers = len(find_decoder_layers(model))
    layer_idx = layer_idx if layer_idx is not None else max(0, min(
        n_layers - 1, int(n_layers * cfg.rmu_layer_frac)))
    window = [i for i in range(max(0, layer_idx - cfg.rmu_window + 1), layer_idx + 1)]
    n_frozen = freeze_all_but_layers(model, window)  # RMU edits a layer block, not the whole adapter
    print(f"RMU: target layer {layer_idx}/{n_layers}, trainable window {window}, "
          f"froze {n_frozen} out-of-window params")

    pool = _Pool(forget_ds, cfg.seed)
    target = _calibrate_target(ref_model, tokenizer, pool, layer_idx, device,
                                cfg.rmu_steer_multiplier, cfg.seed)
    extra = _rmu_extra_loss(ref_model, pool, layer_idx, cfg.retain_weight, target, device)
    return sft(model, tokenizer, retain_ds, cfg.unlearn_epochs, cfg.lr, extra_loss_fn=extra)
