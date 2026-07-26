"""Smoke test for the Tianji arm adaptor node.

Exercises the :class:`unienv_tianji.TianjiArmActor` node **directly**
(constructed with ``connect=False``) without going through ``WorldEnv`` or
``RealWorld``. This is required because the ``unienv`` package published on
PyPI predates a recent local core fix (``RealWorld.reset`` kwargs), so the
node is driven through its lifecycle hooks manually.
"""

import numpy as np
import pytest

import unienv_tianji
from unienv_tianji import TianjiArmActor


def test_tianji_arm_node_smoke():
    node = TianjiArmActor(connect=False)
    try:
        # Static metadata.
        assert TianjiArmActor.n_joints == 7
        assert node.n_joints == 7

        # Action space.
        assert node.action_space is not None
        assert node.action_space.shape == (7,)

        # Observation space keys.
        obs_space = node.observation_space
        expected_keys = {
            "joint_positions",
            "joint_velocities",
            "joint_torques",
        }
        assert expected_keys.issubset(set(obs_space.spaces.keys()))
        # Error / state keys were removed in favour of raising
        # TianjiArmHardwareError (see tests/test_shared_connection.py).
        assert "arm_state" not in obs_space.spaces
        assert "error_code" not in obs_space.spaces
        assert obs_space.spaces["joint_positions"].shape == (7,)

        # Extended proprioception channels are on by default (read from the
        # regular 1kHz feedback frame; no extra blocking calls).
        extended_keys = {
            "joint_currents", "joint_temperatures",
            "joint_friction_estimates", "joint_external_force_estimates",
            "cartesian_force_estimate",
        }
        assert extended_keys <= set(obs_space.spaces.keys())
        assert obs_space.spaces["joint_currents"].shape == (7,)
        assert obs_space.spaces["joint_temperatures"].shape == (7,)
        assert obs_space.spaces["cartesian_force_estimate"].shape == (6,)

        # Before reset, observation is None.
        assert node.get_observation() is None

        # after_reset must populate the observation.
        node.after_reset()
        obs = node.get_observation()
        assert obs is not None
        for key in expected_keys | extended_keys:
            assert key in obs
        assert obs["joint_positions"].shape == (7,)
        assert obs["joint_velocities"].shape == (7,)
        assert obs["joint_torques"].shape == (7,)
        assert obs["joint_currents"].shape == (7,)
        assert obs["joint_temperatures"].shape == (7,)
        assert obs["joint_friction_estimates"].shape == (7,)
        assert obs["joint_external_force_estimates"].shape == (7,)
        assert obs["cartesian_force_estimate"].shape == (6,)
        # Offline -> all extended channels are zeros.
        for k in extended_keys:
            np.testing.assert_array_equal(obs[k], 0.0)
        assert "arm_state" not in obs
        assert "error_code" not in obs

        # set_next_action accepts a correctly-shaped float32 action.
        action = np.zeros((7,), dtype=np.float32)
        node.set_next_action(action)

        # set_next_action rejects a wrong-shape action.
        with pytest.raises(AssertionError):
            node.set_next_action(np.zeros((3,), dtype=np.float32))

        # Lifecycle hooks run without error.
        node.pre_environment_step(0.04)
        node.post_environment_step(0.04)
        node.after_reset()

        # get_observation still returns a populated dict after stepping.
        obs2 = node.get_observation()
        assert obs2 is not None
        assert obs2["joint_positions"].shape == (7,)
    finally:
        # close() must run cleanly even when never connected.
        node.close()


def test_public_api_exposed():
    assert hasattr(unienv_tianji, "TianjiArmActor")
