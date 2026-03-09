"""Small helpers for RL learning-rate scheduling."""


def linear_rampdown_multiplier(it: int, num_steps: int) -> float:
    """Linearly ramp from 1.0 to 0.0 across ``num_steps``, with a defensive floor at 0."""
    lrm = 1.0 - it / num_steps
    return max(0.0, lrm)

