from dataclasses import dataclass, field


@dataclass
class Config:
    base_model: str = "EleutherAI/pythia-410m"  # swap to meta-llama/Llama-3.2-1B if you have the quota
    n_authors: int = 200          # synthetic TOFU-style profiles
    forget_frac: float = 0.1      # fraction of authors held out as the forget set
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lr: float = 1e-4
    finetune_epochs: int = 3      # base-model fine-tune (bakes in the forget set)
    unlearn_epochs: int = 3       # per-arm unlearning steps
    npo_beta: float = 0.1         # NPO temperature
    retain_weight: float = 1.0    # retain-loss regularization weight during unlearning
    rmu_layer_frac: float = 0.35  # target layer as a fraction of depth (paper: early-middle layers)
    rmu_window: int = 3           # number of layers ending at the target layer left trainable
    rmu_steer_multiplier: float = 6.0  # target act. norm = multiplier * avg forget-layer act. norm
    attack_steps: int = 20        # relearning-attack fine-tune steps
    seed: int = 0
    data_dir: str = "data/generated"
    output_dir: str = "results"
