from __future__ import annotations

from dataclasses import replace
from enum import Enum

from .params import ModelParams


class Variant(str, Enum):
    baseline = "baseline"
    feedback = "feedback"
    crosstalk = "crosstalk"
    feedback_crosstalk = "feedback_crosstalk"


def apply_variant(params: ModelParams, variant: Variant) -> ModelParams:
    """Return a new ModelParams with toggles for feedback/crosstalk enabled."""

    if variant == Variant.baseline:
        return params

    if variant == Variant.feedback:
        return replace(
            params,
            k_S6K_to_IRS=max(params.k_S6K_to_IRS, 0.35),
        )

    if variant == Variant.crosstalk:
        return replace(
            params,
            k_ERK_to_TSC=max(params.k_ERK_to_TSC, 0.45),
            k_ERK_to_IRS=max(params.k_ERK_to_IRS, 0.15),
            alpha_AMPK=max(params.alpha_AMPK, 0.75),
        )

    if variant == Variant.feedback_crosstalk:
        return apply_variant(apply_variant(params, Variant.feedback), Variant.crosstalk)

    raise ValueError(f"Unknown variant: {variant}")
