
from __future__ import annotations

import torch.nn as nn

from quartet2.linear import Quartet_II_linear
from prime_rl.utils.logger import get_logger


def replace_linear(model: nn.Module) -> None:
    """Replace all nn.Linear layers with Quartet_II_linear, except the LM head.

    This walks the module tree and swaps every ``nn.Linear`` that is not
    ``model.lm_head`` for a ``Quartet_II_linear`` layer, preserving the
    original weight (and bias, if present).
    """
    logger = get_logger()

    # Collect (parent, attr_name, module) triples for all nn.Linear instances
    # that are *not* the lm_head.
    lm_head = getattr(model, "lm_head", None)
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for parent_name, parent_module in model.named_modules():
        for attr_name, child in parent_module.named_children():
            if isinstance(child, nn.Linear) and child is not lm_head:
                replacements.append((parent_module, attr_name, child))

    count = 0
    for parent, attr_name, old_linear in replacements:
        assert old_linear.bias is None
        new_linear = Quartet_II_linear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            bias=None,
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )
        new_linear.weight = old_linear.weight
        setattr(parent, attr_name, new_linear)
        count += 1

    del old_linear
    logger.info(f"Replaced {count} nn.Linear layers with Quartet_II_linear (lm_head excluded)")
