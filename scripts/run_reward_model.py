"""Common reward training entry point for both experiment groups."""
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from dp3o_alignment import H4ArgumentParser, ModelArguments
from dp3o_alignment.reward_training import (
    RewardConfig, PairwiseRewardCollator, PairwiseRewardTrainer,
    load_reward_dataset, tokenize_reward_pair,
)
from dp3o_alignment.reward_utils import configure_reward_tokenizer


def main():
    model_args, args = H4ArgumentParser((ModelArguments, RewardConfig)).parse()
    if args.target_type not in {"hard", "golden"}:
        raise ValueError("target_type must be hard or golden")
    if model_args.use_peft or model_args.load_in_4bit or model_args.load_in_8bit:
        raise ValueError("This reward entry point uses full fine-tuning; quantization/PEFT are not configured")
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision
    )
    configure_reward_tokenizer(tokenizer, args.max_length)
    required = {"chosen", "rejected"}
    if args.target_type == "golden":
        required |= {"chosen_rewards", "rejected_rewards"}
    processed = []
    for source in (args.train_data, args.val_data):
        data = load_reward_dataset(source)
        missing = required.difference(data.column_names)
        if missing:
            raise ValueError(f"{source} is missing reward training columns {sorted(missing)}")
        if not len(data):
            raise ValueError(f"Empty training/evaluation dataset: {source}")
        with args.main_process_first(desc="Tokenize reward pairs"):
            data = data.map(tokenize_reward_pair,
                            fn_kwargs={"tokenizer": tokenizer, "target_type": args.target_type},
                            num_proc=args.preprocessing_num_workers,
                            remove_columns=data.column_names)
        processed.append(data)
    dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype not in {None, "auto"} else model_args.torch_dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, torch_dtype=dtype,
        revision=model_args.model_revision, trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not args.gradient_checkpointing
    args.label_names = []

    def metrics(prediction):
        chosen, rejected = prediction.predictions
        return {"chosen_accuracy": float(np.mean(chosen > rejected))}

    trainer = PairwiseRewardTrainer(
        model=model, args=args, tokenizer=tokenizer, train_dataset=processed[0].shuffle(seed=args.seed),
        eval_dataset=processed[1], data_collator=PairwiseRewardCollator(tokenizer), compute_metrics=metrics,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if args.do_eval:
        trainer.save_metrics("eval", trainer.evaluate())
    checkpoint = str(Path(args.output_dir) / "last_checkpoint")
    trainer.save_model(checkpoint)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(checkpoint)


if __name__ == "__main__":
    main()
