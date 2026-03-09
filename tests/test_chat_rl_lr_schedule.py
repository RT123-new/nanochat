from nanochat.rl_schedule import linear_rampdown_multiplier


def test_linear_rampdown_multiplier_bounds_and_shape():
    num_steps = 10

    assert linear_rampdown_multiplier(0, num_steps) == 1.0
    assert linear_rampdown_multiplier(5, num_steps) == 0.5
    assert linear_rampdown_multiplier(num_steps, num_steps) == 0.0
    assert linear_rampdown_multiplier(num_steps + 1, num_steps) == 0.0
