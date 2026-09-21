"""Format full preference conversations, including strict Mistral templates."""
from copy import deepcopy

from .data import maybe_insert_system_message


def format_policy_pair(example, tokenizer):
    chosen, rejected = deepcopy(example["chosen"]), deepcopy(example["rejected"])
    if len(chosen) < 2 or len(rejected) < 2:
        raise ValueError("Preference conversations must contain a prompt and an answer")
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        raise ValueError("Both preference conversations must end in an assistant answer")
    if chosen[:-1] != rejected[:-1]:
        raise ValueError("Chosen and rejected conversations must have identical prompts")
    maybe_insert_system_message(chosen, tokenizer)
    maybe_insert_system_message(rejected, tokenizer)
    prompt = tokenizer.apply_chat_template(chosen[:-1], tokenize=False, add_generation_prompt=False)
    result = dict(example)
    result["prompt"] = prompt
    for side, messages in (("chosen", chosen), ("rejected", rejected)):
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError("Chat template does not preserve the prompt prefix in a complete conversation")
        result[side] = full[len(prompt):]
    return result
