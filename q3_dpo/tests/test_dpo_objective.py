import numpy as np
import pytest
from src.dpo_objective import dpo_logits, dpo_loss, preference_accuracy

ARGS = (
    np.array([-2.0, -3.0, -1.5]),
    np.array([-4.0, -2.0, -2.0]),
    np.array([-2.5, -2.5, -1.5]),
    np.array([-3.5, -2.0, -2.5]),
)

def test_logits():
    np.testing.assert_allclose(dpo_logits(*ARGS, beta=0.5), [0.5, -0.25, -0.25])

def test_none():
    expected = np.logaddexp(0.0, -np.array([0.5, -0.25, -0.25]))
    np.testing.assert_allclose(dpo_loss(*ARGS, beta=0.5, reduction="none"), expected)

def test_reductions():
    losses = dpo_loss(*ARGS, beta=0.5, reduction="none")
    assert np.isclose(dpo_loss(*ARGS, beta=0.5, reduction="mean"), np.mean(losses))
    assert np.isclose(dpo_loss(*ARGS, beta=0.5, reduction="sum"), np.sum(losses))

def test_scalar():
    assert np.isclose(dpo_loss(-2.0, -4.0, -2.5, -3.5, beta=0.5), np.logaddexp(0.0, -0.5))

def test_accuracy():
    assert np.isclose(preference_accuracy(*ARGS, beta=0.5), 1 / 3)

def test_invalid_beta():
    with pytest.raises(ValueError):
        dpo_loss([-1.0], [-2.0], [-1.0], [-2.0], beta=0.0)

def test_invalid_reduction():
    with pytest.raises(ValueError):
        dpo_loss([-1.0], [-2.0], [-1.0], [-2.0], beta=0.5, reduction="median")

def test_shapes():
    with pytest.raises(ValueError):
        dpo_loss([-1.0, -2.0], [-2.0], [-1.0], [-2.0], beta=0.5)
