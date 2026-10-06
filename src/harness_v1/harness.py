"""Method-agnostic harness: bake a synthetic forget-set into a base model,
run every registered arm as a plugin, score forget/retain quality + durability
under a simulated relearning attack, write one row per arm to results.csv.
"""
import copy
import csv
import json
import time
from pathlib import Path

from config import Config
from arms.common import load_base_model, add_lora, sft
from arms import REGISTRY
from eval.metrics import forget_retain_quality
from eval.attack import run_attack
from data.make_forget_set import generate


def load_data(cfg):
    data_dir = Path(cfg.data_dir)
    if not (data_dir / "forget.jsonl").exists():
        generate(cfg.n_authors, cfg.forget_frac, cfg.seed, cfg.data_dir)
    forget_ds = [json.loads(l) for l in open(data_dir / "forget.jsonl")]
    retain_ds = [json.loads(l) for l in open(data_dir / "retain.jsonl")]
    return forget_ds, retain_ds


def build_pristine(cfg, tokenizer, forget_ds, retain_ds):
    model, _ = load_base_model(cfg.base_model)
    model = add_lora(model, cfg)
    return sft(model, tokenizer, forget_ds + retain_ds, cfg.finetune_epochs, cfg.lr)


def main(arms=None, cfg=None):
    cfg = cfg or Config()
    arms = arms or list(REGISTRY.keys())
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    forget_ds, retain_ds = load_data(cfg)
    _, tokenizer = load_base_model(cfg.base_model)

    print("Baking pristine model (forget set included)...")
    pristine = build_pristine(cfg, tokenizer, forget_ds, retain_ds)
    pristine_metrics = forget_retain_quality(pristine, tokenizer, forget_ds, retain_ds)
    print("pristine:", pristine_metrics)

    rows = []
    for name in arms:
        kind, run_fn = REGISTRY[name]
        print(f"\n=== Arm: {name} ({kind}) ===")
        t0 = time.time()
        if kind == "from_scratch":
            model = run_fn(cfg.base_model, tokenizer, forget_ds, retain_ds, cfg)
        else:
            model = run_fn(copy.deepcopy(pristine), tokenizer, forget_ds, retain_ds, cfg)
        cost_s = time.time() - t0

        metrics = forget_retain_quality(model, tokenizer, forget_ds, retain_ds)
        attack = run_attack(model, tokenizer, forget_ds, retain_ds, cfg, pristine_metrics["forget_nll"])
        row = {"arm": name, "cost_s": round(cost_s, 1), **metrics, **attack}
        print(row)
        rows.append(row)

    out_path = Path(cfg.output_dir) / "results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")
    return rows


if __name__ == "__main__":
    import sys
    selected = sys.argv[1:] or None  # e.g. `python harness.py retrain npo hru`
    main(arms=selected)
