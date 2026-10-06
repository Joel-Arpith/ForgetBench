"""Arm A: exact unlearning. Two variants of the same idea (never let the
model see the forget set), registered as 'retrain' and 'sisa'.

ponytail: true SISA needs shards fixed and checkpointed from the *original*
training run so only the affected shard is redone. We don't have that
history here (the pristine model was trained in one pass), so 'sisa'
approximates it by sharding the retain set and fine-tuning fresh LoRA
adapters per shard, then averaging — same asymptotic cost story (only
touch retain data, forget shard is simply never trained), not the same
infra. Upgrade to real sharded checkpoints if you need the exact SISA cost
curve rather than just the exact-unlearning guarantee.
"""
import copy
import torch

from .common import load_base_model, add_lora, sft


def run_retrain(base_model_name, tokenizer, forget_ds, retain_ds, cfg):
    model, _ = load_base_model(base_model_name)
    model = add_lora(model, cfg)
    return sft(model, tokenizer, retain_ds, cfg.finetune_epochs, cfg.lr)


def run_sisa(base_model_name, tokenizer, forget_ds, retain_ds, cfg, n_shards=4):
    shards = [retain_ds[i::n_shards] for i in range(n_shards)]
    base, _ = load_base_model(base_model_name)
    shard_models = []
    for shard in shards:
        if not shard:
            continue
        m = add_lora(copy.deepcopy(base), cfg)
        shard_models.append(sft(m, tokenizer, shard, cfg.finetune_epochs, cfg.lr))

    # aggregate: average LoRA adapter weights across shards (SISA's aggregation step)
    final = shard_models[0]
    with torch.no_grad():
        for name, param in final.named_parameters():
            if not param.requires_grad:
                continue
            stacked = torch.stack([dict(m.named_parameters())[name].data for m in shard_models])
            param.data.copy_(stacked.mean(dim=0))
    return final
