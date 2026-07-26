"""Offline tests for the mink-based EEF control node.

All tests run with ``connect=False`` (no hardware). They cover:
* per-arm MJCF asset validity (loads in mujoco, 7 DoF, palm site, finite FK);
* IK convergence + determinism + round-trip FK(IK(T)) ~= T;
* node spaces / observation keys / env-integration (RealWorld + WorldEnv) with
  a dummy obs-only sibling;
* fixed robot-base-frame EEF action/observation semantics;
* the euler convention (R = Rx @ Ry @ Rz) — known-rotation behavior tests,
  since the conversion now delegates to XBArray (single source of truth);
* the rot6d row convention (XBArray), verified via known rotations + a
  hold-pose regression test on both arms;
* lifecycle: reset/reload priority sets declared, reset clears the IK error
  cache + cached action (env end-to-end);
* max_joint_pos_vel dt-aware rate limit (rad/s * dt);
* the pinned synchronous no-cache apply_action action path.
"""

import os
import warnings

import numpy as np
import pytest

# Silence mink's qpsolvers sparse-conversion warnings during tests.
warnings.filterwarnings("ignore", message="Converted matrix")

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld, WorldEnv
from unienv_interface.space import DictSpace, BoxSpace

from unienv_tianji import TianjiArmEefActor
from xbarray.transformations.rotation_conversions.numpy import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "unienv_tianji", "assets"
)
ASSET_DIR = os.path.abspath(ASSET_DIR)


# --------------------------------------------------------------------------- #
# Asset validity
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def mujoco_models():
    import mujoco

    models = {}
    for side, fname in (("left", "tianji_marvin_left_arm_kine.xml"),
                        ("right", "tianji_marvin_right_arm_kine.xml")):
        path = os.path.join(ASSET_DIR, fname)
        m = mujoco.MjModel.from_xml_path(path)
        models[side] = m
    return models


def test_assets_load_and_have_7_dof(mujoco_models):
    import mujoco

    for side, m in mujoco_models.items():
        assert m.nq == 7, f"{side}: nq={m.nq}"
        assert m.nv == 7, f"{side}: nv={m.nv}"
        # palm site exists
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "palm")
        assert sid >= 0, f"{side}: no 'palm' site"
        # home keyframe exists
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert kid >= 0, f"{side}: no 'home' keyframe"


def test_fk_at_home_is_finite_and_arms_differ(mujoco_models):
    import mujoco

    poses = {}
    for side, m in mujoco_models.items():
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "palm")
        pos = d.site_xpos[sid]
        mat = d.site_xmat[sid].reshape(3, 3)
        assert np.all(np.isfinite(pos)), f"{side}: non-finite palm pos"
        assert np.all(np.isfinite(mat)), f"{side}: non-finite palm mat"
        poses[side] = pos.copy()
    # left/right palm frames are mirrored in y (sign flip on y component).
    np.testing.assert_allclose(poses["left"][0], poses["right"][0], atol=1e-3)
    assert abs(poses["left"][1] + poses["right"][1]) < 1e-3, (
        f"left/right palm y not mirrored: {poses['left']} vs {poses['right']}"
    )
    np.testing.assert_allclose(poses["left"][2], poses["right"][2], atol=1e-3)


# --------------------------------------------------------------------------- #
# IK convergence / determinism / round-trip (via the node's mink stack)
# --------------------------------------------------------------------------- #

def _home_qpos(arm):
    from unienv_tianji import REST_JOINT_POSITIONS

    return REST_JOINT_POSITIONS[arm].astype(np.float64)


def _node_fk_palm_root(node, q):
    """Palm pose (4x4) in the MJCF root frame via the node's helpers."""
    pose = node._fk_palm_in_root(q.astype(np.float64))
    return pose


def _assert_eef_obs_matches_root_fk(node, obs):
    """Assert EEF observations are the untransformed MJCF-root palm FK."""
    pose_root = _node_fk_palm_root(node, obs["joint_position"])
    np.testing.assert_allclose(obs["eef_position"], pose_root[:3, 3], atol=1e-6)
    np.testing.assert_allclose(
        quaternion_to_matrix(obs["eef_quaternion"].astype(np.float64)),
        pose_root[:3, :3],
        atol=1e-6,
    )


def test_ik_converges_for_reachable_target():
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        q_home = _home_qpos("A")
        T_home = _node_fk_palm_root(node, q_home)
        # Shift the target -5cm along y (well-conditioned direction).
        T_target = T_home.copy()
        T_target[1, 3] -= 0.05
        q_target, err = node._solve_ik(T_target, q_home)
        assert q_target.shape == (7,)
        assert err.shape == (6,)
        pos_err = np.linalg.norm(err[:3])
        rot_err = np.linalg.norm(err[3:])
        assert pos_err < 1e-3, f"pos_err={pos_err}"
        assert rot_err < 1e-2, f"rot_err={rot_err}"
    finally:
        node.close()


def test_ik_determinism_same_seed_same_target():
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        q_home = _home_qpos("A")
        T_home = _node_fk_palm_root(node, q_home)
        T_target = T_home.copy()
        T_target[2, 3] += 0.05  # +5cm in z
        q1, _ = node._solve_ik(T_target, q_home.copy())
        q2, _ = node._solve_ik(T_target, q_home.copy())
        np.testing.assert_allclose(q1, q2, atol=1e-9)
    finally:
        node.close()


def test_ik_roundtrip_fk_of_ik_matches_target():
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        q_home = _home_qpos("A")
        T_home = _node_fk_palm_root(node, q_home)
        # Several reachable targets: small offsets along y/z and a small rotation.
        targets = []
        T = T_home.copy()
        T[1, 3] -= 0.05
        targets.append(T)
        T = T_home.copy()
        T[2, 3] += 0.05
        targets.append(T)
        T = T_home.copy()
        # Small rotation about z by 0.1 rad.
        th = 0.1
        Rz = np.array([
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th),  np.cos(th), 0],
            [0, 0, 1],
        ], dtype=np.float64)
        T[:3, :3] = Rz @ T[:3, :3]
        targets.append(T)

        for T_target in targets:
            q_target, err = node._solve_ik(T_target, q_home.copy())
            T_achieved = _node_fk_palm_root(node, q_target)
            np.testing.assert_allclose(
                T_achieved[:3, 3], T_target[:3, 3], atol=2e-3,
                err_msg=f"position mismatch: {T_achieved[:3,3]} vs {T_target[:3,3]}"
            )
            # Orientation: compare rotation matrices up to sign (quaternion double cover).
            R_err = T_achieved[:3, :3].T @ T_target[:3, :3]
            # angle of the residual rotation
            cos_th = (np.trace(R_err) - 1) / 2
            cos_th = float(np.clip(cos_th, -1.0, 1.0))
            assert np.arccos(cos_th) < 2e-2, (
                f"orientation mismatch: residual angle={np.arccos(cos_th)}"
            )
    finally:
        node.close()


# --------------------------------------------------------------------------- #
# Node spaces / observation keys / env-integration
# --------------------------------------------------------------------------- #

def test_offline_node_spaces_default_and_rotation_reprs():
    for repr_name, rdim in (("euler", 3), ("quat", 4), ("rot6d", 6)):
        node = TianjiArmEefActor(
            connect=False, arm="A",
            rotation_representation=repr_name,
        )
        try:
            assert node.action_space.shape == (3 + rdim,)
        finally:
            node.close()

    # rot6d via the rot6d path -> 9
    node = TianjiArmEefActor(
        connect=False, arm="A", rotation_representation="rot6d",
    )
    try:
        assert node.action_space.shape == (9,)
    finally:
        node.close()


def test_offline_node_joint_mode_action_space():
    node = TianjiArmEefActor(connect=False, arm="A", action_mode="joint_position")
    try:
        assert node.action_space.shape == (7,)
    finally:
        node.close()


def test_offline_node_obs_keys_exactly():
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        # EEF node's own singular keys + the child's extended-proprioception
        # plural passthrough keys.
        eef_own = {
            "joint_position", "joint_velocity", "joint_torque",
            "eef_position", "eef_quaternion", "last_ik_solved_error",
        }
        extended = {
            "joint_currents", "joint_temperatures",
            "joint_friction_estimates", "joint_external_force_estimates",
            "cartesian_force_estimate",
        }
        expected = eef_own | extended
        assert set(node.observation_space.spaces.keys()) == expected
        node.after_reset()
        obs = node.get_observation()
        assert obs is not None
        assert set(obs.keys()) == expected
        # Base trio + eef keys present with correct shapes.
        assert obs["joint_position"].shape == (7,)
        assert obs["eef_position"].shape == (3,)
        assert obs["eef_quaternion"].shape == (4,)
        assert obs["last_ik_solved_error"].shape == (6,)
        _assert_eef_obs_matches_root_fk(node, obs)
        # Extended passthrough channels (offline zeros).
        assert obs["joint_currents"].shape == (7,)
        assert obs["joint_temperatures"].shape == (7,)
        assert obs["joint_friction_estimates"].shape == (7,)
        assert obs["joint_external_force_estimates"].shape == (7,)
        assert obs["cartesian_force_estimate"].shape == (6,)
        for k in extended:
            np.testing.assert_array_equal(obs[k], 0.0)
        # Before any eef action, last_ik_solved_error is zeros.
        np.testing.assert_allclose(obs["last_ik_solved_error"], 0.0)
    finally:
        node.close()


def test_offline_node_obs_keys_when_child_extended_disabled():
    # When the child actor has read_extended_proprioception=False, the eef
    # node exposes only its own six keys.
    node = TianjiArmEefActor(
        connect=False, arm="A", read_extended_proprioception=False,
    )
    try:
        eef_own = {
            "joint_position", "joint_velocity", "joint_torque",
            "eef_position", "eef_quaternion", "last_ik_solved_error",
        }
        assert set(node.observation_space.spaces.keys()) == eef_own
        node.after_reset()
        obs = node.get_observation()
        assert set(obs.keys()) == eef_own
        _assert_eef_obs_matches_root_fk(node, obs)
    finally:
        node.close()


class _DummyObsNode:
    """An obs-only sibling node: reports a single obs key, no action."""

    after_reset_priorities = {0}
    after_reload_priorities = {0}
    pre_environment_step_priorities = {0}
    post_environment_step_priorities = {0}
    reset_priorities = {0}
    reload_priorities = {0}

    def __init__(self, name="dummy"):
        self.name = name
        self.world = None
        self.control_timestep = 0.04
        self.update_timestep = 0.04
        self.backend = NumpyComputeBackend
        self.device = None
        self.observation_space = DictSpace(NumpyComputeBackend, {
            "counter": BoxSpace(
                NumpyComputeBackend, low=0, high=1e9, dtype=np.float32, shape=(1,),
            ),
        })
        self.action_space = BoxSpace(
            NumpyComputeBackend, low=-1.0, high=1.0, dtype=np.float32, shape=(1,),
        )
        self.context_space = None
        self.has_reward = False
        self.has_termination_signal = False
        self.has_truncation_signal = False
        self._counter = 0

    @property
    def effective_update_timestep(self):
        return self.update_timestep if self.update_timestep is not None else self.control_timestep

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        self._counter = 0

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        self._counter = 0

    def after_reset(self, *, priority=0, mask=None):
        self._counter += 1

    def after_reload(self, *, priority=0, mask=None):
        self._counter += 1

    def pre_environment_step(self, dt, *, priority=0):
        pass

    def post_environment_step(self, dt, *, priority=0):
        self._counter += 1

    def get_observation(self):
        return {"counter": np.array([self._counter], dtype=np.float32)}

    def get_info(self):
        return {}

    def get_context(self):
        return None

    def get_reward(self):
        return 0.0

    def get_termination(self):
        return False

    def get_truncation(self):
        return False

    def set_next_action(self, action):
        # Accepts the per-node sub-action; no-op for this dummy.
        pass

    def close(self):
        pass

    def get_node(self, key):
        if isinstance(key, str) and key == self.name:
            return self
        return None

    def get_nodes_by_fn(self, fn):
        out = []
        if fn(self):
            out.append(self)
        return out

    def get_nodes_by_type(self, t):
        out = []
        if isinstance(self, t):
            out.append(self)
        return out

    def can_render(self):
        return False

    @property
    def render_mode(self):
        return None

    @property
    def supported_render_modes(self):
        return ()


def test_env_integration_offline_eef_hold_pose():
    world = RealWorld(
        NumpyComputeBackend,
        world_timestep=0.04,
        batch_size=None,
    )
    node = TianjiArmEefActor(world=world, connect=False, arm="A")
    dummy = _DummyObsNode("dummy")
    dummy.world = world
    env = WorldEnv(world, [node, dummy])
    try:
        ctx, obs, info = env.reset()
        # obs is nested under node names (CombinedWorldNode with >1 child).
        assert "tianji_arm_eef" in obs
        assert "dummy" in obs
        eef_obs = obs["tianji_arm_eef"]
        for k in ("eef_position", "eef_quaternion", "last_ik_solved_error"):
            assert k in eef_obs
        _assert_eef_obs_matches_root_fk(node, eef_obs)

        # Build a hold-pose eef action from the eef obs.
        R = quaternion_to_matrix(eef_obs["eef_quaternion"].astype(np.float64))
        eul = matrix_to_euler_angles(R, "XYZ")
        action = {
            "tianji_arm_eef": np.concatenate(
                [eef_obs["eef_position"].astype(np.float32), eul.astype(np.float32)]
            ).astype(np.float32),
            "dummy": np.zeros((1,), dtype=np.float32),
        }
        obs2, reward, term, trunc, info = env.step(action)
        assert "tianji_arm_eef" in obs2
        eef2 = obs2["tianji_arm_eef"]
        # Holding the current pose -> error should be tiny.
        err_norm = np.linalg.norm(eef2["last_ik_solved_error"])
        assert err_norm < 1e-2, f"hold-pose err too large: {err_norm}"
    finally:
        env.close()
        node.close()


# --------------------------------------------------------------------------- #
# Euler convention: R = Rx @ Ry @ Rz
# --------------------------------------------------------------------------- #

def test_euler_xyz_convention_matches_formula():
    # Direct formula sanity: a few random euler triples, compare against an
    # explicit Rx @ Ry @ Rz construction.
    rng = np.random.default_rng(0)
    for _ in range(10):
        e = rng.uniform(-np.pi, np.pi, size=3)
        R = euler_angles_to_matrix(e, "XYZ")
        # Build Rx, Ry, Rz manually.
        cx, sx = np.cos(e[0]), np.sin(e[0])
        cy, sy = np.cos(e[1]), np.sin(e[1])
        cz, sz = np.cos(e[2]), np.sin(e[2])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_ref = Rx @ Ry @ Rz
        np.testing.assert_allclose(R, R_ref, atol=1e-10)


def test_euler_xyz_known_rotations():
    """euler_angles_to_matrix("XYZ") == Rx@Ry@Rz for known single-axis and
    composite rotations (behavior-level, now that euler_angles_to_matrix IS
    XBArray, an equivalence test against XBArray would be trivial)."""
    # Pure rotations about each axis.
    for axis, idx, sign in (("X", 0, +1), ("Y", 1, +1), ("Z", 2, +1)):
        for th in (0.0, 0.5, -1.2):
            e = np.zeros(3); e[idx] = th
            R = euler_angles_to_matrix(e, "XYZ")
            # Build the canonical single-axis matrix and compare.
            c, s = np.cos(th), np.sin(th)
            if axis == "X":
                R_ref = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == "Y":
                R_ref = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                R_ref = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            np.testing.assert_allclose(R, R_ref, atol=1e-12)

    # Composite: R = Rx(0.3) @ Ry(0.4) @ Rz(0.5) explicit construction.
    x, y, z = 0.3, 0.4, 0.5
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    np.testing.assert_allclose(euler_angles_to_matrix(np.array([x, y, z]), "XYZ"),
                               Rx @ Ry @ Rz, atol=1e-12)


def test_quaternion_matrix_round_trip():
    """quaternion_to_matrix / matrix_to_quaternion (XBArray) round-trip."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        R = quaternion_to_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        q2 = matrix_to_quaternion(R)
        # Quaternion double cover: q2 == +/- q.
        assert np.allclose(q2, q, atol=1e-9) or np.allclose(q2, -q, atol=1e-9)
        R2 = quaternion_to_matrix(q2)
        np.testing.assert_allclose(R2, R, atol=1e-9)


def test_euler_matrix_to_euler_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(20):
        e = rng.uniform(-np.pi + 0.01, np.pi - 0.01, size=3)
        R = euler_angles_to_matrix(e, "XYZ")
        e2 = matrix_to_euler_angles(R, "XYZ")
        # Euler angles are not unique for a given rotation, so we check the
        # forward map round-trips rather than the angle triple itself.
        np.testing.assert_allclose(euler_angles_to_matrix(e2, "XYZ"), R, atol=1e-9)


# --------------------------------------------------------------------------- #
# rot6d convention: rows (XBArray), verified via known rotations + hold pose
# --------------------------------------------------------------------------- #

def test_rot6d_known_rotations():
    """rot6d round-trips known rotation matrices (behavior-level; now that
    rotation_6d_to_matrix/matrix_to_rotation_6d ARE XBArray, an equivalence
    test against XBArray would be a trivial self-comparison)."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        R = quaternion_to_matrix(q)
        r6d = matrix_to_rotation_6d(R)
        assert r6d.shape == (6,)
        # First two rows of R flatten in row-major order.
        np.testing.assert_allclose(r6d, R[:2, :].reshape(-1), atol=1e-12)
        # Round-trip: rot6d -> matrix recovers R.
        R2 = rotation_6d_to_matrix(r6d)
        np.testing.assert_allclose(R2, R, atol=1e-9)
    # Identity.
    R = np.eye(3)
    r6d = matrix_to_rotation_6d(R)
    np.testing.assert_allclose(r6d, np.array([1, 0, 0, 0, 1, 0]), atol=1e-12)
    np.testing.assert_allclose(rotation_6d_to_matrix(r6d), np.eye(3), atol=1e-12)


def test_rot6d_hold_pose_error_near_zero():
    """A rot6d action built from the node's own eef_quaternion obs must hold
    the current pose (IK target == current pose -> error ~ 0). Both arms.
    """
    for arm in ("A", "B"):
        node = TianjiArmEefActor(
            connect=False, arm=arm, rotation_representation="rot6d",
        )
        try:
            node.after_reset()
            obs = node.get_observation()
            _assert_eef_obs_matches_root_fk(node, obs)
            R = quaternion_to_matrix(obs["eef_quaternion"].astype(np.float64))
            rot6d = matrix_to_rotation_6d(R)
            action = np.concatenate(
                [obs["eef_position"].astype(np.float32), rot6d.astype(np.float32)]
            ).astype(np.float32)
            node.set_next_action(action)
            node.pre_environment_step(0.04)
            node.post_environment_step(0.04)
            obs2 = node.get_observation()
            err = obs2["last_ik_solved_error"]
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err_deg = float(np.degrees(np.linalg.norm(err[3:])))
            assert pos_err < 1e-3, f"arm {arm}: pos_err={pos_err}"
            # The regression this guards against was ~163 deg; require < 1 deg.
            assert rot_err_deg < 1.0, f"arm {arm}: rot_err={rot_err_deg} deg"
        finally:
            node.close()


# --------------------------------------------------------------------------- #
# Lifecycle: reset/reload priorities + IK error cache cleared
# --------------------------------------------------------------------------- #

def test_reset_reload_priorities_declared():
    assert TianjiArmEefActor.reset_priorities == {0}
    assert TianjiArmEefActor.reload_priorities == {0}
    assert TianjiArmEefActor.after_reset_priorities == {0}
    assert TianjiArmEefActor.after_reload_priorities == {0}
    assert TianjiArmEefActor.pre_environment_step_priorities == {0}
    assert TianjiArmEefActor.post_environment_step_priorities == {0}


def test_reset_clears_last_ik_error_and_cached_action():
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        node.after_reset()
        # Produce a nonzero last_ik_solved_error by issuing an unreachable
        # eef action (target far outside the workspace).
        bad_action = np.array(
            [10.0, 10.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        node.set_next_action(bad_action)
        node.pre_environment_step(0.04)
        node.post_environment_step(0.04)
        err = node.get_observation()["last_ik_solved_error"]
        assert not np.allclose(err, 0.0), "precondition: error must be nonzero"

        # reset() must clear the IK error and the cached action.
        node.reset()
        assert np.allclose(node._last_ik_error, 0.0)
        assert node._next_action is None

        # after_reset refreshes obs; last_ik_solved_error is back to zeros.
        node.after_reset()
        obs = node.get_observation()
        np.testing.assert_allclose(obs["last_ik_solved_error"], 0.0)
    finally:
        node.close()


def test_env_reset_clears_ik_error():
    """End-to-end: env.reset() restores last_ik_solved_error to zeros."""
    from unienv_interface.backends.numpy import NumpyComputeBackend
    from unienv_interface.world import RealWorld, WorldEnv

    world = RealWorld(NumpyComputeBackend, world_timestep=0.04, batch_size=None)
    node = TianjiArmEefActor(world=world, connect=False, arm="A")
    dummy = _DummyObsNode("dummy")
    dummy.world = world
    env = WorldEnv(world, [node, dummy])
    try:
        _, _, _ = env.reset()
        # Step with an unreachable eef action to produce a nonzero error.
        bad = {
            "tianji_arm_eef": np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
                                       dtype=np.float32),
            "dummy": np.zeros((1,), dtype=np.float32),
        }
        obs, _, _, _, _ = env.step(bad)
        err = obs["tianji_arm_eef"]["last_ik_solved_error"]
        assert not np.allclose(err, 0.0)

        # reset -> error back to zeros.
        _, obs, _ = env.reset()
        np.testing.assert_allclose(
            obs["tianji_arm_eef"]["last_ik_solved_error"], 0.0
        )
    finally:
        env.close()
        node.close()


# --------------------------------------------------------------------------- #
# max_joint_pos_vel: dt-aware rate limit (rad/s * dt)
# --------------------------------------------------------------------------- #

def test_eef_actor_velocity_defaults():
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        assert node.max_joint_pos_vel == pytest.approx(0.2)
        assert node.arm_actor.vel_ratio == 10
        assert node.arm_actor.acc_ratio == 10
    finally:
        node.close()


def test_eef_actor_velocity_overrides_respected():
    node = TianjiArmEefActor(
        connect=False,
        arm="A",
        max_joint_pos_vel=0.35,
        vel_ratio=14,
        acc_ratio=15,
    )
    try:
        assert node.max_joint_pos_vel == pytest.approx(0.35)
        assert node.arm_actor.vel_ratio == 14
        assert node.arm_actor.acc_ratio == 15
    finally:
        node.close()


def test_max_joint_pos_vel_dt_aware_clip():
    """The per-step |dq| cap is max_joint_pos_vel * dt (not a raw delta)."""
    node = TianjiArmEefActor(
        connect=False, arm="A",
        max_joint_pos_vel=1.0,  # 1 rad/s
    )
    try:
        q_current = np.zeros(7, dtype=np.float64)
        q_target = np.full(7, 10.0, dtype=np.float64)  # huge delta
        # dt=0.04 -> cap 0.04 rad per step.
        clipped = node._clip_joint_delta(q_target, q_current, dt=0.04)
        np.testing.assert_allclose(
            np.abs(clipped - q_current), 0.04, atol=1e-9
        )
        # dt=0.1 -> cap 0.1 rad per step.
        clipped2 = node._clip_joint_delta(q_target, q_current, dt=0.1)
        np.testing.assert_allclose(
            np.abs(clipped2 - q_current), 0.1, atol=1e-9
        )
    finally:
        node.close()


def test_max_joint_pos_vel_none_no_clip():
    node = TianjiArmEefActor(connect=False, arm="A", max_joint_pos_vel=None)
    try:
        q_current = np.zeros(7, dtype=np.float64)
        q_target = np.full(7, 10.0, dtype=np.float64)
        out = node._clip_joint_delta(q_target, q_current, dt=0.04)
        np.testing.assert_array_equal(out, q_target)
    finally:
        node.close()


def test_max_joint_vel_param_removed():
    """The old max_joint_vel parameter must no longer exist."""
    import inspect

    sig = inspect.signature(TianjiArmEefActor.__init__)
    assert "max_joint_pos_vel" in sig.parameters
    assert "max_joint_vel" not in sig.parameters


# --------------------------------------------------------------------------- #
# apply_action: pinned, synchronous, no-cache action path
# --------------------------------------------------------------------------- #

def test_apply_action_eef_offline_updates_ik_error_no_cache():
    """apply_action (eef mode) runs IK and updates _last_ik_error, but does NOT
    populate the cached-action path.
    """
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        node.after_reset()
        before_next = node._next_action
        # Build a hold-pose euler action from obs.
        obs = node.get_observation()
        R = quaternion_to_matrix(obs["eef_quaternion"].astype(np.float64))
        eul = matrix_to_euler_angles(R, "XYZ")
        action = np.concatenate(
            [obs["eef_position"].astype(np.float32), eul.astype(np.float32)]
        ).astype(np.float32)
        node.apply_action(action, dt=0.04)
        # IK ran -> error updated (and ~0 for a hold-pose).
        err = node._last_ik_error
        assert err.shape == (6,)
        assert np.linalg.norm(err) < 1e-2
        # Cached-action path untouched.
        assert node._next_action is before_next
        # A subsequent pre_environment_step with no set_next_action sends
        # nothing extra (no raise, no IK re-run -> error unchanged).
        node.pre_environment_step(0.04)
        np.testing.assert_array_equal(node._last_ik_error, err)
    finally:
        node.close()


def test_apply_action_joint_mode_offline_noop():
    node = TianjiArmEefActor(connect=False, arm="A", action_mode="joint_position")
    try:
        node.apply_action(np.zeros(7, dtype=np.float32), dt=0.04)
        # joint mode never updates the eef IK error.
        np.testing.assert_allclose(node._last_ik_error, 0.0)
        assert node._next_action is None
    finally:
        node.close()


def test_apply_action_bad_shape_raises():
    node = TianjiArmEefActor(connect=False, arm="A", rotation_representation="quat")
    try:
        with pytest.raises(ValueError):
            node.apply_action(np.zeros(6, dtype=np.float32))  # expected 7
    finally:
        node.close()


def test_apply_action_independent_of_set_next_action():
    """apply_action must not interfere with a later cached-action step."""
    node = TianjiArmEefActor(connect=False, arm="A")
    try:
        node.after_reset()
        # Cached action set via set_next_action.
        cached = np.zeros(6, dtype=np.float32)
        node.set_next_action(cached)
        assert node._next_action is cached
        # apply_action with a different action must not overwrite the cache.
        other = np.ones(6, dtype=np.float32)
        node.apply_action(other, dt=0.04)
        assert node._next_action is cached
    finally:
        node.close()
