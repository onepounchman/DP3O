"""Shared tokenizer setup and rendering for sequence-classification rewards."""

REWARD_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token }}{% endif %}{% endfor %}"""


def configure_reward_tokenizer(tokenizer, max_length: int):
    """Set up training; callers must resize embeddings and set model.pad_token_id.

    GPTNeoX pools immediately before the first PAD. EOS also appears between
    chat messages, so using EOS for padding makes the reward ignore the answer.
    """
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        pad_token = "[PAD]" if tokenizer.eos_token != "[PAD]" else "<|reward_pad|>"
        tokenizer.add_special_tokens({"pad_token": pad_token})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = REWARD_CHAT_TEMPLATE
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = max_length
    # These attributes are not all automatically copied into tokenizer_config
    # by Transformers 4.45. Preserve them when the scoring process reloads it.
    tokenizer.init_kwargs.update(
        padding_side=tokenizer.padding_side,
        truncation_side=tokenizer.truncation_side,
        model_max_length=tokenizer.model_max_length,
    )
    return tokenizer


def render_reward_messages(tokenizer, messages):
    """Render the supplied conversation identically for training and scoring."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # The following tokenizer call adds BOS for models that require it. Strip
    # only the leading BOS, preserving any literal BOS tokens in the message.
    if tokenizer.bos_token:
        rendered = rendered.removeprefix(tokenizer.bos_token)
    return rendered


def validate_reward_padding(tokenizer, model_config):
    """Reject legacy GPTNeoX rewards that cannot score the complete answer."""
    if model_config.model_type != "gpt_neox":
        return
    if (
        tokenizer.pad_token_id is None
        or tokenizer.pad_token_id == tokenizer.eos_token_id
        or model_config.pad_token_id != tokenizer.pad_token_id
    ):
        raise ValueError(
            "GPTNeoX reward checkpoints require a dedicated PAD distinct from EOS "
            "and matching model/tokenizer pad_token_id. Retrain the proxy reward "
            "model with the corrected tokenizer, then regenerate proxy scores; "
            "changing the tokenizer of an old checkpoint alone is not sufficient."
        )
