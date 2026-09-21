"""Full-text causal LM labels that preserve real EOS when PAD and EOS share an ID."""
from dataclasses import dataclass

import torch


@dataclass
class AttentionMaskCausalCollator:
    tokenizer: object
    pad_to_multiple_of: int | None = None

    def __call__(self, features):
        if any('labels' in feature for feature in features):
            raise ValueError('This full-text SFT collator expects unlabeled tokenized text')
        batch = dict(self.tokenizer.pad(features, padding=True,
                                       pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='pt'))
        batch.pop('special_tokens_mask', None)
        if 'attention_mask' not in batch:
            raise ValueError('An attention mask is required to distinguish real EOS from padding')
        labels = batch['input_ids'].clone()
        labels.masked_fill_(batch['attention_mask'] == 0, -100)
        batch['labels'] = labels
        return batch


def audit_causal_batch(batch, eos_token_id):
    ids, mask, labels = batch['input_ids'], batch['attention_mask'].bool(), batch['labels']
    if not torch.equal(labels[mask], ids[mask]) or not (labels[~mask] == -100).all():
        raise ValueError('Real tokens were masked or padding tokens were supervised')
    # Causal loss shifts labels: index zero is never a prediction target.
    eos = (ids[:, 1:] == eos_token_id) & mask[:, 1:]
    return dict(real_eos_targets=int(eos.sum()),
                supervised_eos_targets=int(((labels[:, 1:] == eos_token_id) & eos).sum()),
                padding_tokens=int((~mask).sum()),
                supervised_tokens=int(mask[:, 1:].sum()))
