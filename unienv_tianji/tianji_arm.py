import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from unienv_interface.world import WorldNode, RealWorld, World
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import DictSpace, BoxSpace

from .connection import TianjiConnection
from .errors import TianjiArmHardwareError

# Rest (home) joint positions in radians, based on the MJCF "home" keyframe
# in UniEnvOrg/genesis_adaptor (unienv_genesis_collection/robots/
# tianji_marvin_wuji.py: _ARM_HOME_LEFT / _ARM_HOME_RIGHT), with the left-arm
# j5 sign flipped (+0.356, matching the real left arm's convention).
# SDK arm "A" is the left arm and arm "B" the right arm. All fingers open.
REST_JOINT_POSITIONS = {
    "A": np.array([0.2, -0.963, 0.0, -0.85, 0.356, 0.0, 0.0], dtype=np.float32),
    "B": np.array([-0.2, -0.963, 0.0, -0.85, -0.356, 0.0, 0.0], dtype=np.float32),
}
from .sdk.SDK_PYTHON import fx_kine


class TianjiArmActor(WorldNode[
    None, Dict[str, NumpyArrayType], NumpyArrayType,
    NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
]):
    after_reset_priorities = {0}
    after_reload_priorities = {0}
    pre_environment_step_priorities = {0}
    post_environment_step_priorities = {0}

    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    n_joints = len(joint_names)

    # Arm state codes (see fx_robot.Marvin_Robot.set_state docstring).
    ARM_STATE_IDLE = 0          # 下伺服 / disabled — brakes engage and the arm
                                # LOCKS at its current joint positions (verified
                                # on hardware: no gravity sag on disable)
    ARM_STATE_POSITION = 1      # 位置跟随 / position following
    ARM_STATE_PVT = 2           # PVT
    ARM_STATE_TORQ = 3          # 扭矩 / torque
    ARM_STATE_RELEASE = 4       # 协作释放 / collaborative release
    ARM_STATE_ERROR = 100       # error

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "tianji_arm",
        ip: str = "192.168.1.190",
        arm: Literal["A", "B"] = "A",
        *,
        vel_ratio: int = 6,  # 6% of 180 deg/s max = 10.8 deg/s ~= 0.188 rad/s
        acc_ratio: int = 6,
        joint_limit_low_deg: Optional[np.ndarray] = None,
        joint_limit_high_deg: Optional[np.ndarray] = None,
        kine_config_path: Optional[str] = None,
        connect: bool = True,
        connection: Optional[TianjiConnection] = None,
        post_enable_settle: float = 1.0,
        feedback_valid_tol_deg: float = 0.25,
        feedback_valid_frames: int = 5,
        feedback_valid_timeout: float = 10.0,
        control_mode: str = "position",
        joint_kd: Tuple[Tuple, Tuple] = (
            [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],  # K, N·m/deg (<=2/joint)
            [0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2],  # D, N·m/(deg/s) (0..1)
        ),
        cart_kd: Tuple[Tuple, Tuple] = (
            [3000.0, 3000.0, 3000.0, 60.0, 60.0, 60.0, 0.0],  # K: xyz<=3000, rpy<=100, null<=20
            [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0],              # D: ratios 0..1
        ),
        tool_mass: float = 1.40,          # kg (wuji hand + mount, tuned on arm A hardware)
        tool_com: Tuple[float, float, float] = (0.0, 0.0, 80.0),  # mm, flange frame
        tool_inertia: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Ixx,Ixy,Ixz,Iyy,Iyz,Izz
        set_tool_payload: bool = True,
        control_timestep: Optional[float] = 0.04,  # 25Hz
        update_timestep: Optional[float] = 0.04,  # background read/send frequency
        read_extended_proprioception: bool = True,
    ):
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        self.arm = arm
        self.arm_index = 0 if arm == "A" else 1
        self.ip = ip
        self.vel_ratio = vel_ratio
        self.acc_ratio = acc_ratio

        # Control mode: position (rigid following), joint_impedance, or
        # cart_impedance (both torque mode 3 with an impedance sub-mode).
        valid_modes = ("position", "joint_impedance", "cart_impedance")
        if control_mode not in valid_modes:
            raise ValueError(
                f"control_mode must be one of {valid_modes}, got {control_mode!r}"
            )
        self._control_mode = control_mode
        self._joint_kd = (
            [float(x) for x in joint_kd[0]],
            [float(x) for x in joint_kd[1]],
        )
        self._cart_kd = (
            [float(x) for x in cart_kd[0]],
            [float(x) for x in cart_kd[1]],
        )
        self._tool_mass = float(tool_mass)
        self._tool_com = (float(tool_com[0]), float(tool_com[1]), float(tool_com[2]))
        self._tool_inertia = tuple(float(x) for x in tool_inertia)
        self._set_tool_payload = bool(set_tool_payload)
        # Extended proprioception channels (joint currents, temperatures,
        # commanded positions, friction/disturbance estimates, cartesian
        # force estimate) all ride in the SAME regular 1kHz feedback frame
        # (RT_OUT / DCSS subscribe), so they are safe to read every step with
        # no extra blocking/vendor calls. Set False to expose only the basic
        # joint position/velocity/torque trio (matching the pre-extension API).
        self.read_extended_proprioception = bool(read_extended_proprioception)
        # set_tool kineParams (XYZABC, mm/deg) are zero — TCP = flange.
        self._tool_kine_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # dynamicParams = [mass, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        # (COM in MILLIMETERS, per fx_robot.set_tool docstring).
        self._tool_dynamic_params = [
            self._tool_mass,
            self._tool_com[0], self._tool_com[1], self._tool_com[2],
            *self._tool_inertia,
        ]

        # Joint limits in degrees -> radians (stored in radians, like RohandActor).
        if joint_limit_low_deg is None:
            joint_limit_low_deg = np.full(self.n_joints, -180.0, dtype=np.float32)
        if joint_limit_high_deg is None:
            joint_limit_high_deg = np.full(self.n_joints, 180.0, dtype=np.float32)
        self.joint_limit_low_deg = np.asarray(joint_limit_low_deg, dtype=np.float32)
        self.joint_limit_high_deg = np.asarray(joint_limit_high_deg, dtype=np.float32)
        if self.joint_limit_low_deg.shape != (self.n_joints,):
            raise ValueError(f"joint_limit_low_deg must have shape ({self.n_joints},)")
        if self.joint_limit_high_deg.shape != (self.n_joints,):
            raise ValueError(f"joint_limit_high_deg must have shape ({self.n_joints},)")
        self.joint_limit_low = np.deg2rad(self.joint_limit_low_deg)
        self.joint_limit_high = np.deg2rad(self.joint_limit_high_deg)

        # WorldNode-related attributes
        self.name = name
        if isinstance(world, World):
            assert world.backend == NumpyComputeBackend, "World backend must be NumpyComputeBackend."
            assert world.is_control_timestep_compatible(control_timestep), "Control timestep must be a multiple of world timestep."
        self.world = world
        self.control_timestep = control_timestep
        self.update_timestep = update_timestep

        # Shared hardware connection. The vendored SDK only allows ONE
        # connection per process (fixed UDP port + shared-memory DCSS), so
        # dual-arm setups must share a single TianjiConnection. When
        # connect=False the actor runs fully offline (zero observations,
        # no-op commands) and the connection is ignored.
        self.post_enable_settle = post_enable_settle
        self.feedback_valid_tol_deg = feedback_valid_tol_deg
        self.feedback_valid_frames = feedback_valid_frames
        self.feedback_valid_timeout = feedback_valid_timeout
        self._connection: Optional[TianjiConnection] = None
        if connect:
            if connection is None:
                raise ValueError(
                    "TianjiArmActor(connect=True) requires a shared TianjiConnection; "
                    "create one and pass it as connection=. The vendored SDK only "
                    "allows one controller connection per process, so dual-arm setups "
                    "must share a single TianjiConnection."
                )
            if not connection.connected:
                raise ValueError(
                    "TianjiArmActor(connect=True) was given a TianjiConnection that is "
                    "not connected. Call connection.connect() first (or construct it "
                    "with connect=True)."
                )
            self._connection = connection

        # Optional forward-kinematics support.
        self._kine = None
        self._kine_cfg = None
        self._kine_config_path = kine_config_path
        if kine_config_path is not None:
            self._init_kine(kine_config_path)

        # Observation / action spaces.
        obs_spaces: Dict[str, BoxSpace] = {
            "joint_positions": BoxSpace(  # radians, within joint limits
                NumpyComputeBackend,
                low=self.joint_limit_low,
                high=self.joint_limit_high,
                dtype=np.float32,
                shape=(self.n_joints,),
            ),
            "joint_velocities": BoxSpace(  # rad/s, generous bounds
                NumpyComputeBackend,
                low=-2.0 * np.pi,
                high=2.0 * np.pi,
                dtype=np.float32,
                shape=(self.n_joints,),
            ),
            "joint_torques": BoxSpace(  # N·m, sensor torque (m_FB_Joint_SToq)
                NumpyComputeBackend,
                low=-100.0,
                high=100.0,
                dtype=np.float32,
                shape=(self.n_joints,),
            ),
        }
        if self.read_extended_proprioception:
            # All of these ride in the regular 1kHz RT_OUT feedback frame (no
            # extra blocking/vendor calls per step). Units from the SDK
            # collect_data ID table (python_doc_contrl.md §3.1) and the RT_OUT
            # struct comments in fx_robot.py.
            obs_spaces["joint_currents"] = BoxSpace(  # motor current, per-mille (0/00) of rated
                NumpyComputeBackend,
                low=-1000.0, high=1000.0,
                dtype=np.float32, shape=(self.n_joints,),
            )
            obs_spaces["joint_temperatures"] = BoxSpace(  # °C, drive/motor temp
                NumpyComputeBackend,
                low=-40.0, high=150.0,
                dtype=np.float32, shape=(self.n_joints,),
            )
            obs_spaces["joint_friction_estimates"] = BoxSpace(  # N·m, m_EST_Joint_Firc
                NumpyComputeBackend,
                low=-100.0, high=100.0,
                dtype=np.float32, shape=(self.n_joints,),
            )
            obs_spaces["joint_external_force_estimates"] = BoxSpace(  # N·m, m_EST_Joint_Force
                NumpyComputeBackend,
                low=-100.0, high=100.0,
                dtype=np.float32, shape=(self.n_joints,),
            )
            obs_spaces["cartesian_force_estimate"] = BoxSpace(  # N/N·m, m_EST_Cart_FN (6D)
                NumpyComputeBackend,
                low=-200.0, high=200.0,
                dtype=np.float32, shape=(6,),
            )
        if self._kine is not None:
            obs_spaces["tcp_pose"] = BoxSpace(  # 4x4 FK pose matrix
                NumpyComputeBackend,
                low=-1.0e9,
                high=1.0e9,
                dtype=np.float32,
                shape=(4, 4),
            )
        self.observation_space = DictSpace(NumpyComputeBackend, obs_spaces)
        self.action_space = BoxSpace(
            NumpyComputeBackend,
            low=self.joint_limit_low,
            high=self.joint_limit_high,
            dtype=np.float32,
            shape=(self.n_joints,),
        )

        self._current_observation: Optional[Dict[str, NumpyArrayType]] = None
        self._next_action: Optional[NumpyArrayType] = None

        # Enable the arm on the shared connection.
        if connect:
            self._enable_arm_with_idle_on_failure()

    # ========== Backend / Device ==========
    @property
    def backend(self) -> ComputeBackend[NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        return NumpyComputeBackend

    @property
    def device(self) -> None:
        return None

    @property
    def control_mode(self) -> str:
        """Active control mode (``"position"``, ``"joint_impedance"`` or ``"cart_impedance"``)."""
        return self._control_mode

    # ========== Connection / Lifecycle ==========
    def _enable_arm_with_idle_on_failure(self) -> None:
        """Enable the arm, returning it to IDLE if setup fails at any point."""
        try:
            self._enable_arm()
        except BaseException:
            # Never leave hardware in torque mode because setup verification or
            # another enable step failed. Cleanup must not mask the root cause.
            self._best_effort_set_idle()
            raise

    def _best_effort_set_idle(self) -> None:
        """Try to disable the arm without masking an existing failure."""
        connection = self._connection
        if connection is None:
            return
        try:
            with connection.transaction() as r:
                r.set_state(self.arm, self.ARM_STATE_IDLE)
        except BaseException:
            pass

    def _enable_arm(self) -> None:
        """Enable the arm on the shared connection, branching on control_mode.

        Teleop-proven transaction groupings (learned on hardware):
        ``clear_error`` is ALWAYS alone in its own transaction — bundling it
        with ``set_state`` can cause the state command to be dropped. The
        remaining setup is then issued in mode-specific transactions.
        """
        assert self._connection is not None
        mode = self._control_mode

        # txn1: clear_error alone.
        with self._connection.transaction() as r:
            r.clear_error(self.arm)
        time.sleep(0.3)

        if mode == "position":
            # txn2: vel/acc + state together.
            with self._connection.transaction() as r:
                r.set_vel_acc(self.arm, self.vel_ratio, self.acc_ratio)
                r.set_state(self.arm, self.ARM_STATE_POSITION)
            self._wait_for_state(self.ARM_STATE_POSITION, timeout=5.0, poll_interval=0.01)

        elif mode == "joint_impedance":
            # txn2: enter torque mode + joint impedance type + vel/acc.
            with self._connection.transaction() as r:
                r.set_state(self.arm, self.ARM_STATE_TORQ)
                r.set_impedance_type(self.arm, 1)
                r.set_vel_acc(self.arm, self.vel_ratio, self.acc_ratio)
            # txn3: joint impedance KD gains.
            with self._connection.transaction() as r:
                r.set_joint_kd_params(self.arm, list(self._joint_kd[0]), list(self._joint_kd[1]))
            # txn4 (optional): tool payload dynamics.
            if self._set_tool_payload:
                with self._connection.transaction() as r:
                    r.set_tool(self.arm, list(self._tool_kine_params), list(self._tool_dynamic_params))
            self._wait_for_state(self.ARM_STATE_TORQ, timeout=5.0, poll_interval=0.01)
            self._verify_impedance_feedback(1, self._joint_kd, "joint")

        elif mode == "cart_impedance":
            # txn2: cartesian impedance KD gains (must be set before entering
            # torque mode so the controller has gains ready).
            with self._connection.transaction() as r:
                r.set_cart_kd_params(self.arm, list(self._cart_kd[0]), list(self._cart_kd[1]), 2)
            # txn3: enter torque mode + cartesian impedance type + vel/acc.
            with self._connection.transaction() as r:
                r.set_state(self.arm, self.ARM_STATE_TORQ)
                r.set_impedance_type(self.arm, 2)
                r.set_vel_acc(self.arm, self.vel_ratio, self.acc_ratio)
            # txn4 (optional): tool payload dynamics.
            if self._set_tool_payload:
                with self._connection.transaction() as r:
                    r.set_tool(self.arm, list(self._tool_kine_params), list(self._tool_dynamic_params))
            self._wait_for_state(self.ARM_STATE_TORQ, timeout=5.0, poll_interval=0.01)
            self._verify_impedance_feedback(2, self._cart_kd, "cartesian")

        self._wait_for_valid_feedback()
        time.sleep(self.post_enable_settle)

    def _wait_for_valid_feedback(self) -> None:
        """Wait for sustained nonzero joint feedback after enabling.

        Marvin firmware can transiently stream all-zero ``fb_joint_pos`` values
        after an unclean client disconnect. Its ``in_frame_serial`` is unusable
        for detecting this condition, so require several consecutive frames
        with at least one joint outside the zero signature before allowing the
        arm to finish enabling.
        """
        assert self._connection is not None
        timeout = self.feedback_valid_timeout
        deadline = time.monotonic() + timeout
        consecutive_valid = 0
        last_state = None
        last_error = None
        last_feedback = None

        while True:
            data = self._connection.subscribe()
            i = self.arm_index
            state = data["states"][i]
            feedback_deg = np.asarray(data["outputs"][i]["fb_joint_pos"], dtype=float)
            last_state = int(state["cur_state"])
            last_error = int(state["err_code"])
            last_feedback = feedback_deg

            if np.max(np.abs(feedback_deg)) > self.feedback_valid_tol_deg:
                consecutive_valid += 1
                if consecutive_valid >= self.feedback_valid_frames:
                    return
            else:
                consecutive_valid = 0

            if time.monotonic() >= deadline:
                raise TianjiArmHardwareError(
                    self.arm,
                    cur_state=last_state,
                    error_code=last_error,
                    message=(
                        f"Tianji arm {self.arm!r} timed out waiting for valid joint feedback "
                        f"(all-zeros frames) within {timeout}s; last observed "
                        f"fb_joint_pos={last_feedback.tolist()}. This usually means a stale "
                        "controller stream after an unclean disconnect; try a clean reconnect "
                        "or controller power-cycle."
                    ),
                )
            time.sleep(0.03)

    def _verify_impedance_feedback(
        self,
        impedance_type: int,
        kd: Tuple[List[float], List[float]],
        kind: str,
        timeout: float = 1.5,
        poll_interval: float = 0.03,
    ) -> None:
        """Confirm an impedance setup from controller-reported values.

        Current Marvin firmware keeps ``m_InFrameSerial`` fixed at zero, so it
        cannot establish feedback freshness. Instead, poll until all commanded
        values match after :meth:`_wait_for_state` has reached torque mode. A
        frozen buffer can only satisfy this complete-value check if the
        configuration was already applied, which is itself successful.
        """
        assert self._connection is not None
        deadline = time.monotonic() + timeout
        last_observed = None
        expected_k = np.asarray(kd[0], dtype=float)
        expected_d = np.asarray(kd[1], dtype=float)
        while True:
            data = self._connection.subscribe()
            i = self.arm_index
            state = data["states"][i]
            inputs = data["inputs"][i]
            cur_state = int(state["cur_state"])
            err_code = int(state["err_code"])
            if kind == "joint":
                actual_k = np.asarray(inputs["joint_k"], dtype=float)
                actual_d = np.asarray(inputs["joint_d"], dtype=float)
            else:
                actual_k = np.asarray([*inputs["cart_k"], inputs["cart_kn"]], dtype=float)
                actual_d = np.asarray([*inputs["cart_d"], inputs["cart_dn"]], dtype=float)
            actual_impedance_type = int(inputs["imp_type"])
            last_observed = (cur_state, err_code, actual_impedance_type, actual_k, actual_d)
            gains_match = (
                actual_k.shape == expected_k.shape
                and actual_d.shape == expected_d.shape
                # The vendored subscribe wrapper rounds RT_IN values to 4 decimals.
                and np.allclose(actual_k, expected_k, rtol=0.0, atol=1e-4)
                and np.allclose(actual_d, expected_d, rtol=0.0, atol=1e-4)
            )
            if (
                cur_state == self.ARM_STATE_TORQ
                and err_code == 0
                and actual_impedance_type == impedance_type
                and gains_match
            ):
                return
            if time.monotonic() >= deadline:
                assert last_observed is not None
                last_state, last_error, last_imp_type, last_k, last_d = last_observed
                raise TianjiArmHardwareError(
                    self.arm,
                    cur_state=last_state,
                    error_code=last_error,
                    message=(
                        f"Tianji arm {self.arm!r} {kind} impedance verification failed "
                        f"within {timeout}s: expected state={self.ARM_STATE_TORQ}, "
                        f"err_code=0, imp_type={impedance_type}, K={expected_k.tolist()}, "
                        f"D={expected_d.tolist()}; last observed state={last_state}, "
                        f"err_code={last_error}, imp_type={last_imp_type}, "
                        f"K={last_k.tolist()}, D={last_d.tolist()}."
                    ),
                )
            time.sleep(poll_interval)

    def _wait_for_state(self, target_state: int, timeout: float = 5.0, poll_interval: float = 0.01) -> None:
        import time
        deadline = time.monotonic() + timeout
        while True:
            cur_state, err_code = self.get_arm_state()
            if err_code != 0:
                raise RuntimeError(
                    f"Tianji arm {self.arm!r} reported error code {err_code} while waiting for state {target_state}."
                )
            if cur_state == target_state:
                return
            if cur_state == self.ARM_STATE_ERROR:
                raise RuntimeError(
                    f"Tianji arm {self.arm!r} entered error state (cur_state=100) while waiting for state {target_state}."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out after {timeout}s waiting for Tianji arm {self.arm!r} to reach state {target_state} "
                    f"(last cur_state={cur_state})."
                )
            time.sleep(poll_interval)

    # ========== Actor Implementation ==========
    def pre_environment_step(self, dt: float, *, priority: int = 0) -> None:
        if self._next_action is not None:
            self.send_joint_command(self._next_action)

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self._current_observation = self._read_observation()

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.post_environment_step(0.0, priority=priority)

    def after_reload(self, *, priority: int = 0, mask=None) -> None:
        # RohandActor has a known bug where the first WorldEnv.reset uses the
        # reload flow and the initial observation stays None. We fix this by
        # implementing after_reload to delegate to the same refresh logic as
        # after_reset (see after_reload_priorities above).
        self.post_environment_step(0.0, priority=priority)

    def get_observation(self):
        return self._current_observation

    def set_next_action(self, action):
        assert isinstance(action, NumpyArrayType), "Action must be a numpy array."
        assert action.shape == (self.n_joints,), f"Action shape must be ({self.n_joints},), got {action.shape}"
        self._next_action = action

    def close(self):
        """Disable this arm (best-effort) on the shared connection.

        Does NOT close/release the shared :class:`TianjiConnection` — the user
        owns it and may share it with other actors. Safe to call when never
        connected, and robust to a partially-constructed actor (e.g. when the
        constructor raised before the connection was assigned).
        """
        connection = getattr(self, "_connection", None)
        if connection is None:
            return
        try:
            with connection.transaction() as r:
                r.set_state(self.arm, self.ARM_STATE_IDLE)
        except Exception:
            pass
        # Drop our reference; the connection itself is owned by the caller.
        self._connection = None

    # ========== Hardware Read Helpers ==========
    def _read_feedback(self) -> Dict[str, Any]:
        """Subscribe and return the full per-arm feedback frame.

        Returns a dict with the following keys (all numpy float32 unless noted):

        Basic (always read):
            joint_positions     (7,) rad           — m_FB_Joint_Pos (deg → rad)
            joint_velocities    (7,) rad/s         — m_FB_Joint_Vel (deg/s → rad/s)
            joint_torques       (7,) N·m            — m_FB_Joint_SToq (sensor torque)
            cur_state           int                — m_CurState
            err_code            int                — m_ERRCode

        Extended (always read from the same RT_OUT frame; exposed as obs only
        when ``read_extended_proprioception`` is True):
            joint_currents      (7,) per-mille     — m_FB_Joint_CToq (motor current, 0/00 of rated)
            joint_temperatures  (7,) °C             — m_FB_Joint_Them
            joint_friction_estimates (7,) N·m       — m_EST_Joint_Firc
            joint_external_force_estimates (7,) N·m — m_EST_Joint_Force
            cartesian_force_estimate (6,) N/N·m    — m_EST_Cart_FN (6D end-effector force estimate)

        Offline (no connection): all numeric channels are zeros, cur_state=0,
        err_code=0.
        """
        z7 = np.zeros(self.n_joints, dtype=np.float32)
        z6 = np.zeros(6, dtype=np.float32)
        if self._connection is None:
            return {
                "joint_positions": z7,
                "joint_velocities": z7,
                "joint_torques": z7,
                "joint_currents": z7,
                "joint_temperatures": z7,
                "joint_friction_estimates": z7,
                "joint_external_force_estimates": z7,
                "cartesian_force_estimate": z6,
                "cur_state": 0,
                "err_code": 0,
            }
        data = self._connection.subscribe()
        i = self.arm_index
        outputs = data["outputs"][i]
        states = data["states"][i]
        pos_deg = np.asarray(outputs["fb_joint_pos"], dtype=np.float32)
        vel_deg_s = np.asarray(outputs["fb_joint_vel"], dtype=np.float32)
        return {
            "joint_positions": np.deg2rad(pos_deg).astype(np.float32),
            "joint_velocities": np.deg2rad(vel_deg_s).astype(np.float32),
            "joint_torques": np.asarray(outputs["fb_joint_sToq"], dtype=np.float32),
            "joint_currents": np.asarray(outputs["fb_joint_cToq"], dtype=np.float32),
            "joint_temperatures": np.asarray(outputs["fb_joint_them"], dtype=np.float32),
            "joint_friction_estimates": np.asarray(outputs["est_joint_firc"], dtype=np.float32),
            "joint_external_force_estimates": np.asarray(outputs["est_joint_force"], dtype=np.float32),
            "cartesian_force_estimate": np.asarray(outputs["est_cart_fn"], dtype=np.float32),
            "cur_state": int(states["cur_state"]),
            "err_code": int(states["err_code"]),
        }

    def _read_observation(self) -> Dict[str, NumpyArrayType]:
        fb = self._read_feedback()

        if self._connection is not None:
            if fb["cur_state"] == self.ARM_STATE_ERROR or fb["err_code"] != 0:
                raise TianjiArmHardwareError(
                    self.arm, cur_state=fb["cur_state"], error_code=fb["err_code"],
                )

        obs: Dict[str, NumpyArrayType] = {
            "joint_positions": fb["joint_positions"].astype(np.float32),
            "joint_velocities": fb["joint_velocities"].astype(np.float32),
            "joint_torques": fb["joint_torques"].astype(np.float32),
        }
        if self.read_extended_proprioception:
            obs["joint_currents"] = fb["joint_currents"].astype(np.float32)
            obs["joint_temperatures"] = fb["joint_temperatures"].astype(np.float32)
            obs["joint_friction_estimates"] = fb["joint_friction_estimates"].astype(np.float32)
            obs["joint_external_force_estimates"] = fb["joint_external_force_estimates"].astype(np.float32)
            obs["cartesian_force_estimate"] = fb["cartesian_force_estimate"].astype(np.float32)

        if self._kine is not None:
            obs["tcp_pose"] = self._compute_tcp_pose(fb["joint_positions"]).astype(np.float32)

        return obs

    def _compute_tcp_pose(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        """Forward kinematics from current joint positions (radians -> 4x4 pose)."""
        if self._kine is None:
            return np.eye(4, dtype=np.float32)
        joints_deg = np.rad2deg(joint_pos_rad).tolist()
        pose = self._kine.fk(joints_deg)
        if pose is False or pose is None:
            # FK failed; return identity so the observation stays in-space.
            return np.eye(4, dtype=np.float32)
        return np.asarray(pose, dtype=np.float32)

    # ========== Public Helper Methods ==========
    def read_joint_positions(self) -> np.ndarray:
        """Current joint positions in radians (shape (7,))."""
        return self._read_feedback()["joint_positions"]

    def read_joint_velocities(self) -> np.ndarray:
        """Current joint velocities in rad/s (shape (7,))."""
        return self._read_feedback()["joint_velocities"]

    def read_joint_torques(self) -> np.ndarray:
        """Current joint sensor torques in N·m (shape (7,))."""
        return self._read_feedback()["joint_torques"]

    def read_joint_currents(self) -> np.ndarray:
        """Current motor currents in per-mille of rated (shape (7,)).

        From ``m_FB_Joint_CToq`` in the regular feedback frame.
        """
        return self._read_feedback()["joint_currents"]

    def read_joint_temperatures(self) -> np.ndarray:
        """Current joint/drive temperatures in °C (shape (7,)).

        From ``m_FB_Joint_Them`` in the regular feedback frame.
        """
        return self._read_feedback()["joint_temperatures"]

    def read_joint_friction_estimates(self) -> np.ndarray:
        """Joint friction estimates in N·m (shape (7,)).

        From ``m_EST_Joint_Firc`` in the regular feedback frame.
        """
        return self._read_feedback()["joint_friction_estimates"]

    def read_joint_external_force_estimates(self) -> np.ndarray:
        """Joint external-force estimates in N·m (shape (7,)).

        From ``m_EST_Joint_Force`` in the regular feedback frame.
        """
        return self._read_feedback()["joint_external_force_estimates"]

    def read_cartesian_force_estimate(self) -> np.ndarray:
        """End-effector cartesian force estimate (6D: Fx,Fy,Fz,Tx,Ty,Tz).

        Units N / N·m, shape (6,). From ``m_EST_Cart_FN`` in the regular
        feedback frame.
        """
        return self._read_feedback()["cartesian_force_estimate"]

    def send_joint_command(self, positions_rad: np.ndarray) -> None:
        """
        Send a joint position command (radians) to the arm.

        The command is clipped to the configured joint limits, converted to
        degrees, and submitted as a single clear_set / set_joint_cmd_pose /
        send_cmd transaction. It is a no-op when not connected.

        ``set_joint_cmd_pose`` is documented as valid in BOTH position-follow
        (state 1) and torque (state 3) modes, so this is the single motion
        entry point for every control mode.
        """
        positions_rad = np.asarray(positions_rad, dtype=np.float32)
        if positions_rad.shape != (self.n_joints,):
            raise ValueError(f"Expected positions shape ({self.n_joints},), got {positions_rad.shape}")
        if self._connection is None:
            return
        clipped = np.clip(positions_rad, self.joint_limit_low, self.joint_limit_high)
        joints_deg = np.rad2deg(clipped).astype(np.float64).tolist()
        with self._connection.transaction() as r:
            r.set_joint_cmd_pose(self.arm, joints_deg)

    def send_eef_command(self, pose_4x4: np.ndarray) -> None:
        """
        Send an end-effector (TCP) pose command by solving IK and issuing the
        resulting joint targets via :meth:`send_joint_command`.

        The SDK has no streaming cartesian command, so EEF control is
        implemented as IK -> joint targets. The IK is seeded with the current
        joint positions (read from feedback, in degrees as the SDK expects) so
        the solver returns a configuration close to the current one and avoids
        branch jumps. Works in every control mode (the resulting joint targets
        are followed rigidly in position mode, or compliantly in impedance
        modes).

        Parameters
        ----------
        pose_4x4:
            Target 4x4 homogeneous TCP pose (row-major), expressed in the
            arm's SDK base frame: **x forward, y left, z up, millimeters**
            (verified on hardware: at the rest pose the left arm's TCP sits
            at [559.5, +113.4, 252.8] mm, the right arm's mirrored at
            [559.5, -113.4, 252.8] mm). The pose is the TCP/flange pose *in*
            base coordinates, not relative to the flange.

        Raises
        ------
        ValueError
            If ``pose_4x4`` is not shape (4, 4).
        RuntimeError
            If no kinematics helper is configured (pass
            ``kine_config_path="default"`` to the constructor), or if IK
            fails (target unreachable / singular). On IK failure no command is
            sent.
        """
        pose_4x4 = np.asarray(pose_4x4, dtype=np.float64)
        if pose_4x4.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose_4x4.shape}")
        if self._connection is None:
            # Offline: consistent with send_joint_command being a no-op.
            return
        if self._kine is None:
            raise RuntimeError(
                "send_eef_command requires a kinematics helper; construct the "
                "actor with kine_config_path='default' (or a custom .MvKDCfg path)."
            )

        # Seed IK with the current joint configuration (radians -> degrees).
        joint_pos_rad = self._read_feedback()["joint_positions"]
        seed_deg = np.rad2deg(joint_pos_rad).astype(np.float64).tolist()

        sp = fx_kine.FX_InvKineSolvePara()
        sp.set_input_ik_target_tcp(pose_4x4.flatten().tolist())
        sp.set_input_ik_ref_joint(seed_deg)
        sp.set_input_ik_zsp_type(0)  # minimize Euclidean distance to seed
        result = self._kine.ik(sp)
        if result is False or result is None:
            raise RuntimeError(
                f"Tianji arm {self.arm!r}: IK failed for the requested TCP pose "
                "(target unreachable or singular); no command was sent."
            )
        target_joints_deg = result.m_Output_RetJoint.to_list()
        target_joints_rad = np.deg2rad(np.asarray(target_joints_deg, dtype=np.float32))
        self.send_joint_command(target_joints_rad)

    def read_tcp_pose(self) -> np.ndarray:
        """Current TCP pose as a 4x4 matrix (forward kinematics of current joints).

        Expressed in the arm's SDK base frame: x forward, y left, z up,
        millimeters. Returns the identity when offline (no connection) or
        when no kinematics helper is configured.
        """
        if self._connection is None:
            # Offline: no feedback to read and no FK source, mirror
            # _compute_tcp_pose's no-kine fallback to a 4x4 identity.
            return np.eye(4, dtype=np.float32)
        joint_pos = self.read_joint_positions()
        return self._compute_tcp_pose(joint_pos)

    def get_arm_state(self) -> Tuple[int, int]:
        """Return (cur_state, err_code) for this arm."""
        if self._connection is None:
            return 0, 0
        data = self._connection.subscribe()
        states = data["states"][self.arm_index]
        return int(states["cur_state"]), int(states["err_code"])

    def clear_errors(self) -> None:
        """Clear errors on this arm. No-op when not connected."""
        if self._connection is None:
            return
        with self._connection.transaction() as r:
            r.clear_error(self.arm)

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the arm.

        enabled=True re-runs the full :meth:`_enable_arm` sequence so that
        impedance modes re-apply their KD gains, impedance type and tool
        payload. enabled=False sets the arm to IDLE. No-op when not connected.
        """
        if self._connection is None:
            return
        if enabled:
            self._enable_arm_with_idle_on_failure()
        else:
            with self._connection.transaction() as r:
                r.set_state(self.arm, self.ARM_STATE_IDLE)

    # ========== Kinematics ==========
    def _init_kine(self, kine_config_path: str) -> None:
        """Lazily initialize the Marvin_Kine forward-kinematics helper."""
        if kine_config_path == "default":
            kine_config_path = os.path.join(
                os.path.dirname(__file__), "sdk", "ccs_m6_40.MvKDCfg"
            )
        if not os.path.exists(kine_config_path):
            raise FileNotFoundError(f"kine_config_path not found: {kine_config_path}")
        self._kine = fx_kine.Marvin_Kine()
        cfg = self._kine.load_config(arm_type=self.arm_index, config_path=kine_config_path)
        if cfg is None:
            raise RuntimeError(f"Failed to load Tianji kinematics config from {kine_config_path}")
        self._kine_cfg = cfg
        ok = self._kine.initial_kine(
            robot_type=cfg["TYPE"][self.arm_index],
            dh=cfg["DH"][self.arm_index],
            pnva=cfg["PNVA"][self.arm_index],
            j67=cfg["BD"][self.arm_index],
        )
        if not ok:
            raise RuntimeError("Failed to initialize Tianji kinematics parameters.")
