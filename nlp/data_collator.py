"""Prepare the masked-token view of an otherwise unchanged classification batch."""

import torch


def mask_tokens(input_ids, tokenizer, probability):
    """Use BERT's 80% mask / 10% random / 10% unchanged corruption rule."""
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        eligible &= input_ids.ne(token_id)
    selected = eligible & (torch.rand(input_ids.shape, device=input_ids.device) < probability)
    targets = input_ids.masked_fill(~selected, -100)

    replacement = torch.rand(input_ids.shape, device=input_ids.device)
    masked = input_ids.masked_fill(selected & (replacement < 0.8), tokenizer.mask_token_id)
    random_positions = selected & (replacement >= 0.8) & (replacement < 0.9)
    random_ids = torch.randint(len(tokenizer), input_ids.shape, device=input_ids.device)
    return torch.where(random_positions, random_ids, masked), targets
