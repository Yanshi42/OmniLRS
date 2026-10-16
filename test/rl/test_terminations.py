import numpy as np
import pytest

from src.environments_wrappers.rl.terminations import (
    TerminationDebounceState,
    evaluate_termination_v0,
    update_termination_debounce,
)


WHEELS = ("left_wheel", "right_wheel")


def _evaluate(debounce=TerminationDebounceState(), **overrides):
    values = {
        "root_xy_m": np.asarray([1.0, 1.0]),
        "goal_distance_m": 2.0,
        "goal_heading_error_rad": 0.0,
        "episode_policy_steps": 1,
    }
    values.update(overrides)
    return evaluate_termination_v0(debounce=debounce, **values)


def test_wheel_ground_force_is_ignored_but_chassis_force_is_debounced():
    state = update_termination_debounce(
        TerminationDebounceState(),
        classified_contact_forces_by_link={"left_wheel": np.asarray([0.0, 0.0, 100.0])},
        allowed_ground_contact_links=WHEELS,
        roll_rad=0.0,
        pitch_rad=0.0,
    )
    assert state.collision_physics_steps == 0

    for expected in (1, 2):
        state = update_termination_debounce(
            state,
            classified_contact_forces_by_link={"chassis": np.asarray([21.0, 0.0, 0.0])},
            allowed_ground_contact_links=WHEELS,
            roll_rad=0.0,
            pitch_rad=0.0,
        )
        assert state.collision_physics_steps == expected
    assert _evaluate(state).reason == "collision"


def test_collision_counter_resets_below_strict_force_threshold():
    state = update_termination_debounce(
        TerminationDebounceState(collision_physics_steps=1),
        classified_contact_forces_by_link={"chassis": np.asarray([20.0, 0.0, 0.0])},
        allowed_ground_contact_links=WHEELS,
        roll_rad=0.0,
        pitch_rad=0.0,
    )
    assert state.collision_physics_steps == 0


def test_triggered_collision_is_latched_until_episode_reset():
    state = TerminationDebounceState(collision_physics_steps=1)
    state = update_termination_debounce(
        state,
        classified_contact_forces_by_link={"chassis": np.asarray([21.0, 0.0, 0.0])},
        allowed_ground_contact_links=WHEELS,
        roll_rad=0.0,
        pitch_rad=0.0,
    )
    state = update_termination_debounce(
        state,
        classified_contact_forces_by_link={},
        allowed_ground_contact_links=WHEELS,
        roll_rad=0.0,
        pitch_rad=0.0,
    )

    assert state.collision_physics_steps == 0
    assert state.collision_triggered
    assert _evaluate(state).reason == "collision"


def test_rollover_requires_three_consecutive_physics_steps():
    state = TerminationDebounceState()
    for expected in (1, 2, 3):
        state = update_termination_debounce(
            state,
            classified_contact_forces_by_link={},
            allowed_ground_contact_links=WHEELS,
            roll_rad=np.deg2rad(61.0),
            pitch_rad=0.0,
        )
        assert state.rollover_physics_steps == expected
    assert _evaluate(state).reason == "rollover"


def test_termination_priority_collision_before_every_other_reason():
    result = _evaluate(
        TerminationDebounceState(collision_physics_steps=2, rollover_physics_steps=3),
        root_xy_m=np.asarray([0.0, 0.0]),
        goal_distance_m=0.0,
        episode_policy_steps=300,
    )
    assert result.reason == "collision"
    assert result.terminated and not result.truncated and not result.success


def test_out_of_bounds_precedes_success():
    result = _evaluate(root_xy_m=np.asarray([0.44, 1.0]), goal_distance_m=0.0)
    assert result.reason == "out_of_bounds"


def test_success_precedes_time_limit_and_sets_success_flag():
    result = _evaluate(goal_distance_m=0.50, goal_heading_error_rad=np.deg2rad(20.0), episode_policy_steps=300)
    assert result.reason == "success"
    assert result.terminated and not result.truncated and result.success


def test_time_limit_is_truncated_not_terminated():
    result = _evaluate(episode_policy_steps=300)
    assert result.reason == "time_limit"
    assert not result.terminated and result.truncated and not result.success


@pytest.mark.parametrize("x_m", [0.45, 6.04])
def test_out_of_bounds_limits_are_inclusive(x_m):
    assert _evaluate(root_xy_m=np.asarray([x_m, 1.0])).reason is None
