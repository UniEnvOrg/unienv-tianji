"""mink-based end-effector pose control node for the Tianji/Marvin arm.

Wraps one :class:`TianjiArmActor` and mirrors the sim's ``EndEffectorPoseController``
contract (absolute EEF pose action = 3D position + rotation; rotation in
``{"euler"`` intrinsic XYZ, ``"quat"`` wxyz, ``"rot6d"``; optional
``normalize_rotation_action``). IK is solved with `mink
<https://github.com/kevinmks/mink>`_ against a kinematics-only per-arm MJCF,
seeded from the child actor's *current* feedback joint positions (solution
continuity, like the sim's DLS IK). EEF actions and observations are always
expressed directly in the robot base frame, which is the MJCF robot-root frame.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from unienv_interface.world import WorldNode, RealWorld, World
from unienv_interface.backends import BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import (
    NumpyComputeBackend,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
)
from unienv_interface.space import DictSpace, BoxSpace

from .tianji_arm import TianjiArmActor, REST_JOINT_POSITIONS
from .errors import TianjiArmHardwareError

# --------------------------------------------------------------------------- #
# Rotation-representation utilities.
#
# Action rotation parsing (euler/quat/rot6d) and the EEF quaternion observation
# use XBArray's numpy-backend rotation conversions directly (called at the use
# sites, no pass-through wrappers): ``euler_angles_to_matrix(angles, convention)``,
# ``quaternion_to_matrix``, ``matrix_to_quaternion``, ``rotation_6d_to_matrix``.
# XBArray is the single source of truth for rotation math shared with the sim
# stack, so the real adaptor and the sim cannot drift. Quaternions are
# ``(w, x, y, z)`` (wxyz) both in XBArray and here.
#
# Semantics (verified against the sim):
#   - euler: ``euler_angles_to_matrix(a, "XYZ")`` == R = Rx@Ry@Rz.
#   - rot6d: XBArray row convention (first two rows of R -> 6d; 6d -> b1,b2,b3
#     stacked as rows along axis=-2).
#
# XBArray's numpy backend handles float32 inputs natively (output dtype follows
# input), so no float64 coercion is applied at the call sites.
# --------------------------------------------------------------------------- #

from xbarray.transformations.rotation_conversions.numpy import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

RotationRepr = Literal["euler", "quat", "rot6d"]
_EULER_CONVENTION = "XYZ"


def unnormalize_rotation_action(representation: RotationRepr, action: np.ndarray) -> np.ndarray:
    """Undo the actor's rotation-action normalization."""
    action = np.asarray(action, dtype=np.float64)
    if representation == "euler":
        action = action * np.pi  # [-1, 1] -> [-pi, pi]
    elif representation == "quat":
        n = np.clip(np.linalg.norm(action, axis=-1, keepdims=True), 1e-6, None)
        action = action / n
    return action


def rot_dim(representation: RotationRepr) -> int:
    return {"euler": 3, "quat": 4, "rot6d": 6}[representation]


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_ARM_XML = {
    "A": "tianji_marvin_left_arm_kine.xml",
    "B": "tianji_marvin_right_arm_kine.xml",
}
_PALM_SITE = "palm"


@lru_cache(maxsize=None)
def _rest_eef_pose_cached(arm: Literal["A", "B"]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the canonical home palm pose from the per-arm kinematics MJCF."""
    import mujoco

    xml_path = os.path.join(_ASSET_DIR, _ARM_XML[arm])
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[:] = REST_JOINT_POSITIONS[arm]
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _PALM_SITE)
    if site_id < 0:
        raise RuntimeError(f"Kinematics MJCF {xml_path} has no {_PALM_SITE!r} site")
    position = data.site_xpos[site_id].copy()
    rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    return position, matrix_to_quaternion(rotation)


def rest_eef_pose(arm: Literal["A", "B"]) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical home palm position and wxyz quaternion for one arm."""
    position, quaternion = _rest_eef_pose_cached(arm)
    return position.copy(), quaternion.copy()


class TianjiArmEefActor(WorldNode[
    None, Dict[str, NumpyArrayType], NumpyArrayType,
    NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
]):
    """End-effector pose controller node for one Tianji/Marvin arm.

    Wraps a :class:`TianjiArmActor` (constructed internally or supplied) and
    exposes an absolute end-effector-pose action space (3D position + rotation)
    matching the sim's ``EndEffectorPoseController`` contract. IK is solved with
    `mink <https://github.com/kevinmks/mink>`_ against a kinematics-only per-arm
    MJCF, seeded each step from the child actor's *current* feedback joint
    positions (solution continuity, like the sim's DLS IK). The resulting joint
    targets are sent to the child via :meth:`send_joint_command` (never
    :meth:`set_next_action`, to avoid the stale-resend hazard).

    Observations (singular keys matching the sim): ``joint_position(7)``,
    ``joint_velocity(7)``, ``joint_torque(7)``, ``eef_position(3)``,
    ``eef_quaternion(4 wxyz)``, ``last_ik_solved_error(6) = [pos_err, rot_err]``.
    The EEF pose is the per-arm MJCF palm-frame FK, reported directly in the
    robot base frame (== MJCF robot-root frame); there is no action/world-frame
    transform — actions and observations are ALWAYS expressed in the robot base
    frame. This matches the sim env, which anchors the robot at the world
    origin so its world frame coincides with the robot base frame and its EEF
    actions are world-frame. Teleop/task-frame transforms are the
    responsibility of future retargeting nodes, not this actor.

    Parameters
    ----------
    arm:
        ``"A"`` (left) or ``"B"`` (right). Selects the per-arm MJCF and the child
        actor's arm id.
    action_mode:
        ``"eef_pose"`` (default) exposes the absolute EEF pose action space; IK
        is solved each step. ``"joint_position"`` passes the action through to
        the child actor (a 7-vector of joint targets); EEF observations are
        still provided via FK, and ``last_ik_solved_error`` stays zero.
    rotation_representation:
        ``"euler"`` (intrinsic XYZ, ``R = Rx @ Ry @ Rz``), ``"quat"`` (wxyz) or
        ``"rot6d"``.
    normalize_rotation_action:
        If True, the action's rotation part is normalized (euler to ``[-1, 1]``
        i.e. radians / pi; quat to unit length) before being un-normalized for
        IK. Mirrors the sim option of the same name.
    """

    after_reset_priorities = {0}
    after_reload_priorities = {0}
    pre_environment_step_priorities = {0}
    post_environment_step_priorities = {0}
    reset_priorities = {0}
    reload_priorities = {0}

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "tianji_arm_eef",
        *,
        arm: Literal["A", "B"] = "A",
        ip: str = "192.168.1.190",
        connection: Optional[Any] = None,
        connect: bool = True,
        arm_actor: Optional[TianjiArmActor] = None,
        action_mode: Literal["eef_pose", "joint_position"] = "eef_pose",
        rotation_representation: RotationRepr = "euler",
        normalize_rotation_action: bool = False,
        ik_max_iters: int = 50,
        ik_pos_tol: float = 1e-3,
        ik_ori_tol: float = 1e-2,
        ik_damping: float = 1e-3,
        ik_solver: str = "osqp",
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        lm_damping: float = 0.0,
        control_timestep: Optional[float] = 0.04,
        update_timestep: Optional[float] = 0.04,
        **arm_actor_kwargs: Any,
    ):
        # ``close`` may be called by WorldNode.__del__ if construction fails.
        self.arm_actor: Optional[TianjiArmActor] = None
        self._owns_arm_actor = False
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        if action_mode not in ("eef_pose", "joint_position"):
            raise ValueError(
                f"action_mode must be 'eef_pose' or 'joint_position', got {action_mode!r}"
            )
        if rotation_representation not in ("euler", "quat", "rot6d"):
            raise ValueError(
                f"rotation_representation must be 'euler', 'quat' or 'rot6d', "
                f"got {rotation_representation!r}"
            )
        self.arm = arm
        self.action_mode = action_mode
        self.rotation_representation = rotation_representation
        self.normalize_rotation_action = normalize_rotation_action

        # IK config.
        self.ik_max_iters = int(ik_max_iters)
        self.ik_pos_tol = float(ik_pos_tol)
        self.ik_ori_tol = float(ik_ori_tol)
        self.ik_damping = float(ik_damping)
        self.ik_solver = ik_solver
        self.position_cost = float(position_cost)
        self.orientation_cost = float(orientation_cost)
        self.lm_damping = float(lm_damping)

        # WorldNode bookkeeping.
        self.name = name
        if isinstance(world, World):
            assert world.backend == NumpyComputeBackend, "World backend must be NumpyComputeBackend."
            assert world.is_control_timestep_compatible(control_timestep), \
                "Control timestep must be a multiple of world timestep."
        self.world = world
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep

        # Child actor (owned if we constructed it).
        self._owns_arm_actor = arm_actor is None
        if arm_actor is None:
            # The SDK's velocity and acceleration ratios are the only real-arm
            # velocity limit. Explicit caller overrides win.
            arm_actor_kwargs.setdefault("vel_ratio", 10)
            arm_actor_kwargs.setdefault("acc_ratio", 10)
            arm_actor = TianjiArmActor(
                world=world,
                name=(name + "_arm" if name else "tianji_arm"),
                ip=ip,
                arm=arm,
                connect=connect,
                connection=connection,
                control_timestep=control_timestep,
                update_timestep=update_timestep,
                **arm_actor_kwargs,
            )
        else:
            if arm_actor.arm != arm:
                raise ValueError(
                    f"arm_actor.arm ({arm_actor.arm!r}) != arm ({arm!r})"
                )
        self.arm_actor = arm_actor
        self.n_joints = arm_actor.n_joints

        # Lazily-loaded mink model/configuration (heavy imports kept local).
        self._mujoco_model = None
        self._mink_configuration = None
        self._frame_task = None
        self._limits = None
        self._ik_dt = float(control_timestep) if control_timestep else 0.04

        # Observation cache.
        self._current_observation: Optional[Dict[str, NumpyArrayType]] = None
        self._last_ik_error = np.zeros(6, dtype=np.float32)
        self._next_action: Optional[NumpyArrayType] = None

        # Build spaces.
        n = self.n_joints
        obs_spaces: Dict[str, BoxSpace] = {
            "joint_position": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf, high=np.inf, dtype=np.float32, shape=(n,),
            ),
            "joint_velocity": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf, high=np.inf, dtype=np.float32, shape=(n,),
            ),
            "joint_torque": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf, high=np.inf, dtype=np.float32, shape=(n,),
            ),
            "eef_position": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,),
            ),
            "eef_quaternion": BoxSpace(
                NumpyComputeBackend,
                low=-1.0, high=1.0, dtype=np.float32, shape=(4,),
            ),
            "last_ik_solved_error": BoxSpace(
                NumpyComputeBackend,
                low=-np.inf, high=np.inf, dtype=np.float32, shape=(6,),
            ),
        }
        # Pass-through of the child (TianjiArmActor) extended-proprioception
        # channels, when the child exposes them. These keep the child's plural
        # adaptor names (downstream maps them); the EEF node's own
        # joint_position/velocity/torque stay singular.
        if getattr(self.arm_actor, "read_extended_proprioception", False):
            child_spaces = self.arm_actor.observation_space.spaces
            for k in (
                "joint_currents",
                "joint_temperatures",
                "joint_friction_estimates",
                "joint_external_force_estimates",
                "cartesian_force_estimate",
            ):
                if k in child_spaces:
                    obs_spaces[k] = child_spaces[k]
        self.observation_space = DictSpace(NumpyComputeBackend, obs_spaces)

        rdim = rot_dim(rotation_representation)
        if action_mode == "eef_pose":
            self.action_space = BoxSpace(
                NumpyComputeBackend,
                low=-np.inf, high=np.inf, dtype=np.float32, shape=(3 + rdim,),
            )
        else:
            self.action_space = BoxSpace(
                NumpyComputeBackend,
                low=self.arm_actor.joint_limit_low,
                high=self.arm_actor.joint_limit_high,
                dtype=np.float32, shape=(n,),
            )

    # ========== Backend / Device ==========
    @property
    def backend(self):
        return NumpyComputeBackend

    @property
    def device(self):
        return None

    # ========== MJCF / mink lazy init ==========
    def _load_mink(self):
        """Lazily load the per-arm MJCF and create the mink configuration/task."""
        if self._mujoco_model is not None:
            return
        import mujoco
        import mink

        xml_path = os.path.join(_ASSET_DIR, _ARM_XML[self.arm])
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"Per-arm MJCF not found: {xml_path}. Run "
                "scripts/generate_kine_mjcf.py to regenerate the assets."
            )
        model = mujoco.MjModel.from_xml_path(xml_path)
        self._mujoco_model = model
        self._mink_configuration = mink.Configuration(model)
        self._frame_task = mink.FrameTask(
            frame_name=_PALM_SITE,
            frame_type="site",
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            gain=1.0,
            lm_damping=self.lm_damping,
        )
        self._limits = [mink.ConfigurationLimit(model=model)]

    def _set_configuration(self, q_rad: np.ndarray) -> None:
        """Update the mink configuration from a radians joint vector."""
        import mujoco

        cfg = self._mink_configuration
        # MJCF is in radians (compiler angle="radian"), so qpos == joint rad.
        cfg.update(q=np.asarray(q_rad, dtype=np.float64))

    def _fk_palm_in_root(self, q_rad: np.ndarray) -> np.ndarray:
        """Palm-frame pose (4x4) in the MJCF robot-root frame."""
        import mink

        self._load_mink()
        self._set_configuration(q_rad)
        T = self._mink_configuration.get_transform_frame_to_world(_PALM_SITE, "site")
        # SE3 wxyz_xyz -> 4x4.
        R = quaternion_to_matrix(T.wxyz_xyz[:4])
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = T.wxyz_xyz[4:]
        return M

    # ========== IK ==========
    def _solve_ik(self, target_pose_root_4x4: np.ndarray, seed_q_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve IK for an absolute palm target pose (MJCF root frame, metres).

        Iterates ``solve_ik`` + ``integrate`` from the seed configuration until
        the task error is below tolerance or ``ik_max_iters`` is reached. Does
        NOT raise on non-convergence — the best solution and its error are
        returned (the error goes to ``last_ik_solved_error``).

        Returns
        -------
        q_target : (7,) target joint positions (radians).
        error : (6,) [pos_err(3), rot_err(3)] from the last iterate (axis-angle
            of ``inv(achieved) @ target`` for the rotation part).
        """
        import mink

        self._load_mink()
        cfg = self._mink_configuration
        task = self._frame_task

        # Seed from the current configuration (solution continuity).
        self._set_configuration(seed_q_rad)

        # Build the target SE3 (root frame, metres).
        R = target_pose_root_4x4[:3, :3]
        q_wxyz = matrix_to_quaternion(R)
        xyz = target_pose_root_4x4[:3, 3]
        target = mink.SE3(wxyz_xyz=np.concatenate([q_wxyz, xyz]))
        task.set_target(target)

        q = np.asarray(seed_q_rad, dtype=np.float64).copy()
        err = np.zeros(6, dtype=np.float64)
        for _ in range(self.ik_max_iters):
            v = mink.solve_ik(
                cfg, [task], self._ik_dt, self.ik_solver,
                damping=self.ik_damping, limits=self._limits,
            )
            q = cfg.integrate(v, self._ik_dt)
            cfg.update(q)
            err = task.compute_error(cfg)
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err = float(np.linalg.norm(err[3:]))
            if pos_err <= self.ik_pos_tol and rot_err <= self.ik_ori_tol:
                break
        return q.astype(np.float64), err.astype(np.float64)

    # ========== Action resolution (shared by pre_environment_step / apply_action) ==========
    def _resolve_eef_action(self, action: np.ndarray) -> None:
        """Resolve an eef_pose action into a joint command and send it.

        Parses position + rotation, solves IK seeded from the child's current
        feedback joints, updates ``_last_ik_error``, and dispatches through
        ``child.send_joint_command``.
        Does NOT touch the cached ``_next_action`` state.
        """
        action = np.asarray(action, dtype=np.float64)
        target_pos_root = action[:3]
        target_rot = action[3:]
        if self.normalize_rotation_action:
            target_rot = unnormalize_rotation_action(self.rotation_representation, target_rot)
        target_quat_root = matrix_to_quaternion(
            _rotation_to_matrix(self.rotation_representation, target_rot)
        )

        # Actions are absolute palm poses in the MJCF robot-root/base frame.
        T_root = np.eye(4, dtype=np.float64)
        T_root[:3, :3] = quaternion_to_matrix(target_quat_root)
        T_root[:3, 3] = target_pos_root

        # Seed from the child's CURRENT feedback joint positions.
        q_current = self.arm_actor.read_joint_positions().astype(np.float64)
        q_target, err = self._solve_ik(T_root, q_current)
        self._last_ik_error = err.astype(np.float32)

        # Send to child (never set_next_action — stale-resend hazard).
        self.arm_actor.send_joint_command(q_target.astype(np.float32))

    def apply_action(self, action, dt: Optional[float] = None) -> None:
        """Synchronous, NO-CACHE action path (pinned for downstream use).

        Resolves ``action`` and dispatches it to the child immediately, WITHOUT
        touching any cached / next-action state — this method is fully
        independent of :meth:`set_next_action` / :meth:`pre_environment_step`.
        ``tianji_sim_real`` calls this directly.

        - eef mode: parse per ``rotation_representation`` (+ normalize), solve
          IK seeded from current feedback joints, update ``_last_ik_error``, and
          send via ``child.send_joint_command``.
        - joint mode: ``child.send_joint_command(action)`` directly.

        Parameters
        ----------
        action:
            Action vector matching the configured action space. For eef mode
            this is ``[pos(3), rot(rot_dim)]``; for joint mode ``q_target(7)``.
        dt:
            Reserved for lifecycle compatibility and ignored.
        """
        action = np.asarray(action, dtype=np.float32)
        if self.action_mode == "joint_position":
            self.arm_actor.send_joint_command(action)
            return
        # eef_pose mode: validate shape (lenient on dtype).
        expected = 3 + rot_dim(self.rotation_representation)
        if action.shape != (expected,):
            raise ValueError(
                f"apply_action: expected shape ({expected},), got {action.shape} "
                f"(action_mode={self.action_mode!r}, "
                f"rotation_representation={self.rotation_representation!r})"
            )
        self._resolve_eef_action(action.astype(np.float64))

    # ========== WorldNode lifecycle ==========
    def pre_environment_step(self, dt: float, *, priority: int = 0) -> None:
        action = self._next_action
        if action is None:
            return
        if self.action_mode == "joint_position":
            # Passthrough to child (clipped by the child's own limits).
            self.arm_actor.send_joint_command(np.asarray(action, dtype=np.float32))
            # No IK was run, so the last-error stays as-is (zeros before first
            # eef action; the joint path never updates it).
            return
        self._resolve_eef_action(np.asarray(action, dtype=np.float64))

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self._current_observation = self._read_observation()

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.post_environment_step(0.0, priority=priority)

    def after_reload(self, *, priority: int = 0, mask=None) -> None:
        self.post_environment_step(0.0, priority=priority)

    def reset(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs) -> None:
        # Reset IK-result cache and cached action (mirrors sim controller
        # resetting its IK result cache), then forward to the child actor.
        self._last_ik_error = np.zeros(6, dtype=np.float32)
        self._next_action = None
        self.arm_actor.reset(priority=priority, seed=seed, mask=mask, **kwargs)

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs) -> None:
        self._last_ik_error = np.zeros(6, dtype=np.float32)
        self._next_action = None
        self.arm_actor.reload(priority=priority, seed=seed, mask=mask, **kwargs)

    def get_observation(self):
        return self._current_observation

    def set_next_action(self, action) -> None:
        action = np.asarray(action, dtype=np.float32)
        if self.action_mode == "eef_pose":
            expected = 3 + rot_dim(self.rotation_representation)
        else:
            expected = self.n_joints
        if action.shape != (expected,):
            raise ValueError(
                f"Action shape must be ({expected},), got {action.shape} "
                f"(action_mode={self.action_mode!r}, "
                f"rotation_representation={self.rotation_representation!r})"
            )
        self._next_action = action

    def close(self) -> None:
        """Close the child actor only if this node constructed it.

        Never closes the shared :class:`TianjiConnection` (the caller owns it),
        consistent with :class:`TianjiArmActor`.
        """
        if getattr(self, "_owns_arm_actor", False):
            arm_actor = getattr(self, "arm_actor", None)
            if arm_actor is not None:
                arm_actor.close()

    # ========== Observation ==========
    def _read_observation(self) -> Dict[str, NumpyArrayType]:
        # Child feedback (raises TianjiArmHardwareError on a fault); a single
        # subscribe covers all proprioception channels.
        child_obs = self.arm_actor._read_observation()
        joint_pos = child_obs["joint_positions"].astype(np.float32)

        # EEF pose via FK, reported directly in the MJCF robot-root/base frame.
        pose_root = self._fk_palm_in_root(joint_pos.astype(np.float64))
        eef_pos = pose_root[:3, 3].astype(np.float32)
        eef_quat = matrix_to_quaternion(pose_root[:3, :3]).astype(np.float32)

        obs: Dict[str, NumpyArrayType] = {
            "joint_position": joint_pos,
            "joint_velocity": child_obs["joint_velocities"].astype(np.float32),
            "joint_torque": child_obs["joint_torques"].astype(np.float32),
            "eef_position": eef_pos,
            "eef_quaternion": eef_quat,
            "last_ik_solved_error": self._last_ik_error.astype(np.float32),
        }
        # Pass-through of the child's extended-proprioception channels (plural
        # adaptor names), when the child exposes them.
        if getattr(self.arm_actor, "read_extended_proprioception", False):
            for k in (
                "joint_currents",
                "joint_temperatures",
                "joint_friction_estimates",
                "joint_external_force_estimates",
                "cartesian_force_estimate",
            ):
                if k in child_obs:
                    obs[k] = child_obs[k].astype(np.float32)
        return obs


def _rotation_to_matrix(representation: RotationRepr, rotation: np.ndarray) -> np.ndarray:
    if representation == "euler":
        return euler_angles_to_matrix(rotation, _EULER_CONVENTION)
    elif representation == "quat":
        return quaternion_to_matrix(rotation)
    elif representation == "rot6d":
        return rotation_6d_to_matrix(rotation)
    raise ValueError(f"Unsupported rotation representation: {representation!r}")
