"""Offline tests for the space metadata of the Tianji arm nodes (connect=False)."""

import warnings

import pytest

warnings.filterwarnings("ignore", message="Converted matrix")

from unienv_interface.space.space_utils.metadata_utils import validate_metadata

from unienv_tianji import TianjiArmActor, TianjiArmEefActor

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")


@pytest.mark.parametrize("extended", [True, False])
@pytest.mark.parametrize("control_mode", ["position", "joint_impedance", "cart_impedance"])
def test_arm_actor_spaces_are_described(extended, control_mode):
    actor = TianjiArmActor(
        arm="A", name="left", connect=False, control_mode=control_mode,
        read_extended_proprioception=extended,
    )
    validate_metadata(actor.observation_space)
    validate_metadata(actor.action_space)

    obs = actor.observation_space.spaces
    expected_units = {"joint_positions": "rad", "joint_velocities": "rad/s", "joint_torques": "N*m"}
    if extended:
        expected_units.update({
            "joint_currents": "permille",
            "joint_temperatures": "degC",
            "joint_friction_estimates": "N*m",
            "joint_external_force_estimates": "N*m",
        })
    for key, units in expected_units.items():
        md = obs[key].metadata
        assert md["kind"] == "proprio"
        assert md["units"] == units
        assert tuple(md["names"]) == JOINT_NAMES
        assert md["source"] == "left"
        assert md["fps"] == 25.0
    if extended:
        wrench = obs["cartesian_force_estimate"].metadata
        assert wrench["kind"] == "force"
        assert tuple(wrench["units"]) == ("N", "N", "N", "N*m", "N*m", "N*m")
    else:
        assert "cartesian_force_estimate" not in obs

    # Every control mode commands absolute joint positions.
    act = actor.action_space.metadata
    assert act["kind"] == "action"
    assert act["control"] == "joint_position"
    assert act["mode"] == "absolute"
    assert act["units"] == "rad"
    assert tuple(act["names"]) == JOINT_NAMES


def test_fps_follows_control_timestep():
    actor = TianjiArmActor(arm="B", connect=False, control_timestep=0.1)
    assert actor.observation_space["joint_positions"].metadata["fps"] == 10.0
    actor = TianjiArmActor(arm="B", connect=False, control_timestep=None)
    assert "fps" not in actor.observation_space["joint_positions"].metadata


@pytest.mark.parametrize(
    "representation, normalize, rotation, rot_units",
    [
        ("euler", False, "euler_rxyz", ("rad",) * 3),
        ("euler", True, "euler_rxyz", ("normalized",) * 3),
        ("quat", False, "quat_wxyz", ("normalized",) * 4),
        ("rot6d", False, "rot6d", ("normalized",) * 6),
    ],
)
def test_eef_node_spaces_are_described(representation, normalize, rotation, rot_units):
    node = TianjiArmEefActor(
        arm="A", name="left_arm", connect=False,
        rotation_representation=representation, normalize_rotation_action=normalize,
    )
    validate_metadata(node.observation_space)
    validate_metadata(node.action_space)

    obs = node.observation_space.spaces
    for key, units in (("joint_position", "rad"), ("joint_velocity", "rad/s"), ("joint_torque", "N*m")):
        assert obs[key].metadata["units"] == units
        assert tuple(obs[key].metadata["names"]) == JOINT_NAMES
    assert obs["eef_position"].metadata["frame"] == "base"
    assert obs["eef_position"].metadata["units"] == "m"
    assert obs["eef_quaternion"].metadata["rotation"] == "quat_wxyz"
    assert obs["eef_quaternion"].metadata["frame"] == "base"
    assert tuple(obs["last_ik_solved_error"].metadata["units"]) == ("m",) * 3 + ("rad",) * 3
    # Pass-through channels keep the child's description, attributed to this node.
    assert obs["joint_currents"].metadata["units"] == "permille"
    for space in obs.values():
        assert space.metadata["source"] == "left_arm"
        assert space.metadata["fps"] == 25.0

    act = node.action_space.metadata
    assert act["control"] == "eef_pose"
    assert act["mode"] == "absolute"
    assert act["frame"] == "base"
    assert act["rotation"] == rotation
    assert tuple(act["units"]) == ("m", "m", "m") + rot_units
    assert len(act["names"]) == node.action_space.shape[-1]


def test_eef_node_joint_action_is_described():
    node = TianjiArmEefActor(arm="B", connect=False, action_mode="joint_position")
    validate_metadata(node.action_space)
    act = node.action_space.metadata
    assert act["control"] == "joint_position"
    assert act["mode"] == "absolute"
    assert act["units"] == "rad"
    assert tuple(act["names"]) == JOINT_NAMES
