"""Pairwise reward fitting for hard preferences or golden-score distillation."""
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments

from .reward_utils import render_reward_messages


@dataclass
class RewardConfig(TrainingArguments):
    train_data: str = ""
    val_data: str = ""
    target_type: str = "hard"
    max_length: int = 4096
    preprocessing_num_workers: int = 4
    remove_unused_columns: bool = False


def load_reward_dataset(source):
    path = Path(source)
    if path.is_dir():
        return load_from_disk(str(path))
    if path.suffix.lower() in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path), split="train")
    raise ValueError(f"Expected a prepared dataset directory or JSON file: {source}")


def scalar_score(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Each golden reward must be a scalar or a one-element list")
        value = value[0]
    value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("Golden rewards must be finite")
    return value


def tokenize_reward_pair(example, tokenizer, target_type):
    result = {}
    for key in ("chosen", "rejected"):
        encoded = tokenizer(render_reward_messages(tokenizer, example[key]), truncation=True)
        for field in ("input_ids", "attention_mask"):
            result[f"{key}_{field}"] = encoded[field]
    if target_type == "golden":
        delta = scalar_score(example["chosen_rewards"]) - scalar_score(example["rejected_rewards"])
        result["target_probability"] = torch.sigmoid(torch.tensor(delta)).item()
    elif target_type == "hard":
        result["target_probability"] = 1.0
    else:
        raise ValueError(f"Unknown reward target: {target_type}")
    return result


@dataclass
class PairwiseRewardCollator:
    tokenizer: object

    def __call__(self, features):
        examples = [{field: row[f"{side}_{field}"] for field in ("input_ids", "attention_mask")}
                    for row in features for side in ("chosen", "rejected")]
        # BatchEncoding.to assumes every value is a tensor. A plain dictionary
        # lets Accelerate move tensors while retaining the return_loss boolean.
        batch = dict(self.tokenizer.pad(examples, padding=True, return_tensors="pt"))
        batch["target_probability"] = torch.tensor([r["target_probability"] for r in features])
        batch["return_loss"] = True
        return batch


class PairwiseRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        scores = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        chosen, rejected = scores[0::2, 0], scores[1::2, 0]
        target = inputs["target_probability"].to(chosen.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits((chosen - rejected).float(), target)
        if return_outputs:
            return loss, {"chosen_rewards": chosen, "rejected_rewards": rejected}
        return loss
