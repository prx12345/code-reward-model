"""CodeBERT over [problem] </s> [code]. Head-tail truncation, checkpoint/resume."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import config
from src.utils import log

CHECKPOINT_NAME = "codebert_checkpoint.pt"
BEST_NAME = "codebert_best.pt"


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

@dataclass
class EncodedBatch:
    input_ids: Any
    attention_mask: Any
    labels: Any


def head_tail_truncate(token_ids: list[int], budget: int,
                       head_fraction: float = 0.6) -> tuple[list[int], bool]:
    """Keep the start and the end of a token sequence, drop the middle."""
    if len(token_ids) <= budget:
        return token_ids, False
    head = int(budget * head_fraction)
    tail = budget - head
    return token_ids[:head] + token_ids[-tail:], True


class PairEncoder:
    """Encodes (problem, code) into CodeBERT inputs.

    Problem gets 128 tokens, code gets the rest. Naive longest_first
    truncation would let a long HumanEval docstring eat the code.
    """

    def __init__(self, tokenizer: Any, max_length: int | None = None,
                 problem_budget: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_LENGTH
        self.problem_budget = problem_budget or config.PROBLEM_TOKEN_BUDGET
        self.n_special = tokenizer.num_special_tokens_to_add(pair=True)
        self.stats = {"n": 0, "code_truncated": 0, "problem_truncated": 0}

    def encode(self, problem: str, code: str) -> list[int]:
        tok = self.tokenizer
        problem_ids = tok.encode(problem, add_special_tokens=False)
        code_ids = tok.encode(code, add_special_tokens=False)

        problem_ids, problem_cut = head_tail_truncate(problem_ids, self.problem_budget,
                                                      head_fraction=1.0)
        code_budget = self.max_length - self.n_special - len(problem_ids)
        code_ids, code_cut = head_tail_truncate(code_ids, max(1, code_budget))

        self.stats["n"] += 1
        self.stats["code_truncated"] += int(code_cut)
        self.stats["problem_truncated"] += int(problem_cut)
        return tok.build_inputs_with_special_tokens(problem_ids, code_ids)

    def truncation_report(self) -> dict[str, Any]:
        n = max(1, self.stats["n"])
        return {
            "n_encoded": self.stats["n"],
            "code_truncated": self.stats["code_truncated"],
            "code_truncated_pct": round(100.0 * self.stats["code_truncated"] / n, 2),
            "problem_truncated": self.stats["problem_truncated"],
            "problem_truncated_pct": round(100.0 * self.stats["problem_truncated"] / n, 2),
            "max_length": self.max_length,
            "problem_token_budget": self.problem_budget,
        }


def build_dataset(rows: list[dict[str, Any]], encoder: PairEncoder) -> Any:
    import torch
    from torch.utils.data import Dataset

    class _Rows(Dataset):
        def __init__(self) -> None:
            self.items = [
                (encoder.encode(r["problem_text"], r["code"]), int(r["label"]))
                for r in rows
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            ids, label = self.items[index]
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.float)}

    return _Rows()


def make_collate(pad_token_id: int) -> Any:
    import torch

    def collate(batch: list[dict[str, Any]]) -> EncodedBatch:
        lengths = [len(item["input_ids"]) for item in batch]
        width = max(lengths)
        input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for i, item in enumerate(batch):
            n = lengths[i]
            input_ids[i, :n] = item["input_ids"]
            attention[i, :n] = 1
        labels = torch.stack([item["label"] for item in batch])
        return EncodedBatch(input_ids, attention, labels)

    return collate


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _save_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any,
                     scaler: Any, step: int, epoch: int, best_auc: float) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": step,
        "epoch": epoch,
        "best_auc": best_auc,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": {"model": config.CLASSIFIER_MODEL, "lr": config.LEARNING_RATE,
                   "epochs": config.EPOCHS, "seed": config.SEED,
                   "max_length": config.MAX_LENGTH},
    }, tmp)
    tmp.replace(path)


def predict(model: Any, loader: Any, device: str) -> np.ndarray:
    import torch

    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            ).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def train_classifier(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]],
                     pos_weight: float, resume: bool = True) -> dict[str, Any]:
    """Fine-tune CodeBERT. Returns training history plus artefact paths."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from src.utils import set_seed

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"training on {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
    encoder = PairEncoder(tokenizer)
    train_dataset = build_dataset(train_rows, encoder)
    train_report = encoder.truncation_report()

    val_encoder = PairEncoder(tokenizer)
    val_dataset = build_dataset(val_rows, val_encoder)

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                              shuffle=True, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE,
                            shuffle=False, collate_fn=collate)

    # num_labels=1 + BCEWithLogits rather than num_labels=2 + CrossEntropy:
    # a single logit is what `pos_weight` expects, and it is the score we want
    # for AUC anyway.
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_MODEL, num_labels=1
    ).to(device)

    decay_params = [p for n, p in model.named_parameters()
                    if not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay_params = [p for n, p in model.named_parameters()
                       if any(k in n for k in ("bias", "LayerNorm.weight"))]
    optimizer = AdamW([
        {"params": decay_params, "weight_decay": config.WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.LEARNING_RATE)

    total_steps = max(1, len(train_loader) * config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.WARMUP_RATIO), total_steps
    )
    use_amp = config.FP16 and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    checkpoint_path = config.MODELS_DIR / CHECKPOINT_NAME
    best_path = config.MODELS_DIR / BEST_NAME
    start_epoch, global_step, best_auc = 0, 0, -1.0
    history: list[dict[str, Any]] = []

    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        best_auc = float(state["best_auc"])
        torch.set_rng_state(state["torch_rng"].cpu() if hasattr(state["torch_rng"], "cpu")
                            else state["torch_rng"])
        log(f"resumed from {checkpoint_path.name}: epoch {start_epoch}, "
            f"step {global_step}, best_auc {best_auc:.4f}")

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        running_loss, n_batches = 0.0, 0
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch.input_ids.to(device)
            attention = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids,
                                   attention_mask=attention).logits.squeeze(-1)
                    loss = criterion(logits.float(), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids=input_ids,
                               attention_mask=attention).logits.squeeze(-1)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1
            global_step += 1

            if global_step % config.CHECKPOINT_EVERY_STEPS == 0:
                _save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                                 scaler, global_step, epoch, best_auc)
            if global_step % 50 == 0:
                log(f"epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {running_loss / max(1, n_batches):.4f}")

        val_scores = predict(model, val_loader, device)
        y_val = np.array([r["label"] for r in val_rows], dtype=int)
        val_auc = float(roc_auc_score(y_val, val_scores)) if len(set(y_val)) > 1 else float("nan")
        entry = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, n_batches), 4),
            "val_auc": round(val_auc, 4),
            "seconds": round(time.time() - epoch_start, 1),
        }
        history.append(entry)
        log(f"epoch {epoch} done: {entry}")

        if math.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auc": val_auc}, best_path)
            log(f"new best val AUC {best_auc:.4f} -> {best_path.name}")
        _save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                         global_step, epoch + 1, best_auc)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False)["model"])

    return {
        "history": history,
        "best_val_auc": round(best_auc, 4),
        "train_truncation": train_report,
        "val_truncation": val_encoder.truncation_report(),
        "best_checkpoint": str(best_path),
        "device": device,
        "total_steps": total_steps,
        "_model": model,
        "_tokenizer": tokenizer,
    }


def score_rows(model: Any, tokenizer: Any, rows: list[dict[str, Any]],
               device: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Score arbitrary rows with a trained model."""
    from torch.utils.data import DataLoader

    encoder = PairEncoder(tokenizer)
    dataset = build_dataset(rows, encoder)
    loader = DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
                        collate_fn=make_collate(tokenizer.pad_token_id))
    return predict(model, loader, device), encoder.truncation_report()
