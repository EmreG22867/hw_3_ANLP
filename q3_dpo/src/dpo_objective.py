

from __future__ import annotations

import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=float)


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"All log-probability inputs must have the same shape, got {shapes}")


def dpo_logits(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
):
    if beta <= 0:
        raise ValueError("beta must be positive")

    policy_chosen_logps = _as_array(policy_chosen_logps)
    policy_rejected_logps = _as_array(policy_rejected_logps)
    ref_chosen_logps = _as_array(ref_chosen_logps)
    ref_rejected_logps = _as_array(ref_rejected_logps)
    _validate_same_shape(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    return beta * (policy_logratios - ref_logratios)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
    reduction: str = "mean",
):
    """Compute `-log(sigmoid(DPO logit))` with a stable softplus identity."""
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: 'none', 'mean', or 'sum'")

    logits = dpo_logits(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )
    losses = np.logaddexp(0.0, -logits)

    if reduction == "none":
        return losses
    if reduction == "mean":
        return float(np.mean(losses))
    return float(np.sum(losses))


def preference_accuracy(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float = 0.1,
) -> float:
    logits = dpo_logits(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )
    return float(np.mean(logits > 0))
