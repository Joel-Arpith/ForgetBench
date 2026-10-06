"""Shared model/data plumbing used by every arm and by eval/*.
Keeping this in one place is what makes harness.py able to treat
retrain/SISA/NPO/RMU/HRU as interchangeable plugins.
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def qa_to_text(qa):
    return f"Q: {qa['question']}\nA: {qa['answer']}"


def load_base_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def add_lora(model, cfg):
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def _batch_loss(model, tokenizer, texts, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    out = model(**enc, labels=labels)
    return out.loss


def sft(model, tokenizer, qas, epochs, lr, batch_size=8, device=None, extra_loss_fn=None):
    """Plain supervised fine-tuning on qa_to_text(qa) strings.
    extra_loss_fn(model, tokenizer, batch_qas, device) -> scalar tensor added to the CE loss,
    used by NPO/RMU/HRU to inject their unlearning objective without duplicating the loop.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loader = DataLoader(qas, batch_size=batch_size, shuffle=True, collate_fn=list)

    for _ in range(epochs):
        for batch in loader:
            texts = [qa_to_text(qa) for qa in batch]
            loss = _batch_loss(model, tokenizer, texts, device)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model, tokenizer, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def per_example_avg_nll(model, tokenizer, qas, device):
    """Grad-enabled per-example mean token NLL (batch,). Used as the -logprob
    term in NPO; caller wraps in torch.no_grad() for a frozen reference model."""
    texts = [qa_to_text(qa) for qa in qas]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    out = model(**enc, labels=None)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    mask = (tgt != -100)
    tgt_safe = tgt.clone()
    tgt_safe[~mask] = 0
    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask
    return -(tok_logp.sum(dim=1) / mask.sum(dim=1).clamp(min=1))


def _resolve_layers_path(model):
    """Best-effort locate the transformer block list across common HF causal-LM
    architectures (Llama/Mistral: model.layers, GPT-NeoX/Pythia: gpt_neox.layers,
    GPT-2 family: transformer.h). Returns (dotted_path, module_list)."""
    candidates = ["model.layers", "gpt_neox.layers", "transformer.h"]
    candidates += [f"base_model.{p}" for p in candidates]  # peft-wrapped models
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return path, obj
        except AttributeError:
            continue
    raise ValueError("Unrecognized architecture — add its layer path to _resolve_layers_path().")


def find_decoder_layers(model):
    return _resolve_layers_path(model)[1]


def freeze_all_but_layers(model, layer_indices):
    """RMU forgets by editing a small block of layers, not the whole adapter —
    that's what keeps the edit localized instead of degrading everything.
    Freezes every trainable (LoRA) param outside the given layer indices."""
    path, _ = _resolve_layers_path(model)
    leaf = path.split(".")[-1]  # "layers" or "h"
    tokens = [f".{leaf}.{i}." for i in layer_indices]
    n_frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad and not any(tok in name for tok in tokens):
            p.requires_grad_(False)
            n_frozen += 1
    return n_frozen


@torch.no_grad()
def avg_answer_nll(model, tokenizer, qas, device=None, batch_size=8):
    """Mean per-token negative log-likelihood of the answer span. Lower = model
    is more confident / 'remembers' this fact better. Used for both forget-quality
    (want high after unlearning) and retain-quality (want low / unchanged)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    total, count = 0.0, 0
    for i in range(0, len(qas), batch_size):
        batch = qas[i:i + batch_size]
        texts = [qa_to_text(qa) for qa in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = model(**enc, labels=labels)
        total += out.loss.item() * enc["attention_mask"].sum().item()
        count += enc["attention_mask"].sum().item()
    return total / max(count, 1)
