"""Composite reward function placeholder for PPO training.

The real implementation will blend BERTScore, SARI, FKGL readability, and an
entity preservation penalty. SageMaker training jobs can import this module and
call ``compute_reward`` once the metrics are fully implemented. Until then we
return neutral values that make the PPO loop easy to wire up without touching
control flow.
"""

from __future__ import annotations

from typing import Dict


DEFAULT_WEIGHTS: Dict[str, float] = {
    "bertscore": 0.5,
    "sari": 0.3,
    "readability": 0.2,
    "entity_penalty": 1.0,
}


def compute_reward(*, bertscore: float = 0.0, sari: float = 0.0, readability_bonus: float = 0.0, penalty: float = 0.0) -> float:
    """Return a weighted reward using placeholder inputs."""

    reward = (
        DEFAULT_WEIGHTS["bertscore"] * bertscore
        + DEFAULT_WEIGHTS["sari"] * sari
        + DEFAULT_WEIGHTS["readability"] * readability_bonus
        - DEFAULT_WEIGHTS["entity_penalty"] * penalty
    )
    return reward


__all__ = ["compute_reward", "DEFAULT_WEIGHTS"]
