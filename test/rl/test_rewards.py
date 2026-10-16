import numpy as np
import pytest

from src.environments_wrappers.rl.rewards import RewardV0Input, reward_v0_components


def test_reward_v0_golden_component_values():
    components = reward_v0_components(
        RewardV0Input(
            previous_distance_m=0.9,
            distance_m=0.8,
            previous_heading_error_rad=np.pi / 2.0,
            heading_error_rad=np.pi / 4.0,
            previous_applied_action=np.zeros(2),
            applied_action=np.asarray([1.0, -1.0]),
            roll_rad=np.deg2rad(45.0),
            pitch_rad=0.0,
            termination_reason="success",
        )
    )

    assert components == pytest.approx(
        {
            "progress": 0.5,
            "heading": 0.125,
            "action": -0.04,
            "tilt": -0.05,
            "time": -0.01,
            "terminal": 10.0,
            "total": 10.525,
        }
    )


def test_reward_progress_is_clipped_and_heading_is_near_goal_only():
    components = reward_v0_components(
        RewardV0Input(
            previous_distance_m=3.0,
            distance_m=4.0,
            previous_heading_error_rad=1.0,
            heading_error_rad=0.0,
            previous_applied_action=np.zeros(2),
            applied_action=np.zeros(2),
            roll_rad=0.0,
            pitch_rad=0.0,
            termination_reason="time_limit",
        )
    )

    assert components["progress"] == pytest.approx(-1.25)
    assert components["heading"] == 0.0
    assert components["terminal"] == 0.0


@pytest.mark.parametrize("reason", ["collision", "rollover", "out_of_bounds"])
def test_reward_v0_failure_reasons_have_frozen_terminal_penalty(reason):
    components = reward_v0_components(
        RewardV0Input(
            previous_distance_m=2.0,
            distance_m=2.0,
            previous_heading_error_rad=0.0,
            heading_error_rad=0.0,
            previous_applied_action=np.zeros(2),
            applied_action=np.zeros(2),
            roll_rad=0.0,
            pitch_rad=0.0,
            termination_reason=reason,
        )
    )
    assert components["terminal"] == -10.0


def test_reward_v0_rejects_non_finite_inputs():
    with pytest.raises(ValueError, match="finite"):
        reward_v0_components(
            RewardV0Input(
                previous_distance_m=np.nan,
                distance_m=1.0,
                previous_heading_error_rad=0.0,
                heading_error_rad=0.0,
                previous_applied_action=np.zeros(2),
                applied_action=np.zeros(2),
                roll_rad=0.0,
                pitch_rad=0.0,
            )
        )
