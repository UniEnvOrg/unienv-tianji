import os
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from unienv_interface.world import WorldNode, RealWorld, World
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import DictSpace, BoxSpace

from .sdk import fx_robot, fx_kine


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
    ARM_STATE_IDLE = 0          # 下伺服 / disabled
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
        vel_ratio: int = 10,
        acc_ratio: int = 10,
        joint_limit_low_deg: Optional[np.ndarray] = None,
        joint_limit_high_deg: Optional[np.ndarray] = None,
        kine_config_path: Optional[str] = None,
        connect: bool = True,
        control_timestep: Optional[float] = 0.04,  # 25Hz
        update_timestep: Optional[float] = 0.04,  # background read/send frequency
    ):
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        self.arm = arm
        self.arm_index = 0 if arm == "A" else 1
        self.ip = ip
        self.vel_ratio = vel_ratio
        self.acc_ratio = acc_ratio

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

        # Hardware handles (only created when connect=True).
        self._robot = None
        self._dcss = None
        self._connected = False

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
            "joint_torques": BoxSpace(  # N·m, sensor torque
                NumpyComputeBackend,
                low=-100.0,
                high=100.0,
                dtype=np.float32,
                shape=(self.n_joints,),
            ),
            "arm_state": BoxSpace(  # cur_state (0..255)
                NumpyComputeBackend,
                low=0.0,
                high=255.0,
                dtype=np.float32,
                shape=(1,),
            ),
            "error_code": BoxSpace(  # err_code (0..2**31)
                NumpyComputeBackend,
                low=0.0,
                high=float(2 ** 31),
                dtype=np.float32,
                shape=(1,),
            ),
        }
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

        # Connect to hardware.
        if connect:
            self._connect_and_enable()

    # ========== Backend / Device ==========
    @property
    def backend(self) -> ComputeBackend[NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        return NumpyComputeBackend

    @property
    def device(self) -> None:
        return None

    # ========== Connection / Lifecycle ==========
    def _connect_and_enable(self) -> None:
        """Connect to the controller, clear errors, set vel/acc and enter position mode."""
        self._robot = fx_robot.Marvin_Robot()
        self._dcss = fx_robot.DCSS()
        ok = self._robot.connect(self.ip)
        if not ok:
            raise ConnectionError(
                f"Failed to connect to Tianji/Marvin robot at {self.ip}. "
                "Ensure the controller is reachable (ping) and the network cable is plugged in."
            )
        self._connected = True
        try:
            self.clear_errors()
            self._robot.set_vel_acc(self.arm, self.vel_ratio, self.acc_ratio)
            self._robot.set_state(self.arm, self.ARM_STATE_POSITION)
            # Wait (bounded) for the arm to actually reach position-following state.
            self._wait_for_state(self.ARM_STATE_POSITION, timeout=5.0, poll_interval=0.01)
        except Exception:
            # Best-effort cleanup before propagating the failure.
            try:
                self._robot.release_robot()
            except Exception:
                pass
            self._connected = False
            self._robot = None
            self._dcss = None
            raise

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
        if not self._connected or self._robot is None:
            return
        try:
            self._robot.clear_set()
            self._robot.set_state(self.arm, self.ARM_STATE_IDLE)
            self._robot.send_cmd()
        except Exception:
            pass
        try:
            self._robot.release_robot()
        except Exception:
            pass
        self._connected = False
        self._robot = None
        self._dcss = None

    # ========== Hardware Read Helpers ==========
    def _read_feedback(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Subscribe and return (joint_pos_rad, joint_vel_rad_s, joint_torque_Nm, cur_state, err_code)."""
        if not self._connected or self._robot is None or self._dcss is None:
            zeros = np.zeros(self.n_joints, dtype=np.float32)
            return zeros, zeros, zeros, 0, 0
        data = self._robot.subscribe(self._dcss)
        i = self.arm_index
        outputs = data["outputs"][i]
        states = data["states"][i]
        pos_deg = np.asarray(outputs["fb_joint_pos"], dtype=np.float32)
        vel_deg_s = np.asarray(outputs["fb_joint_vel"], dtype=np.float32)
        stoq = np.asarray(outputs["fb_joint_sToq"], dtype=np.float32)
        joint_pos = np.deg2rad(pos_deg).astype(np.float32)
        joint_vel = np.deg2rad(vel_deg_s).astype(np.float32)
        cur_state = int(states["cur_state"])
        err_code = int(states["err_code"])
        return joint_pos, joint_vel, stoq, cur_state, err_code

    def _read_observation(self) -> Dict[str, NumpyArrayType]:
        joint_pos, joint_vel, joint_torque, cur_state, err_code = self._read_feedback()

        if self._connected:
            if cur_state == self.ARM_STATE_ERROR or err_code != 0:
                raise RuntimeError(
                    f"Tianji arm {self.arm!r} entered an error state (cur_state={cur_state}, err_code={err_code})."
                )

        obs: Dict[str, NumpyArrayType] = {
            "joint_positions": joint_pos.astype(np.float32),
            "joint_velocities": joint_vel.astype(np.float32),
            "joint_torques": joint_torque.astype(np.float32),
            "arm_state": np.asarray([cur_state], dtype=np.float32),
            "error_code": np.asarray([err_code], dtype=np.float32),
        }

        if self._kine is not None:
            obs["tcp_pose"] = self._compute_tcp_pose(joint_pos).astype(np.float32)

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
        joint_pos, _, _, _, _ = self._read_feedback()
        return joint_pos

    def read_joint_velocities(self) -> np.ndarray:
        """Current joint velocities in rad/s (shape (7,))."""
        _, joint_vel, _, _, _ = self._read_feedback()
        return joint_vel

    def read_joint_torques(self) -> np.ndarray:
        """Current joint sensor torques in N·m (shape (7,))."""
        _, _, joint_torque, _, _ = self._read_feedback()
        return joint_torque

    def send_joint_command(self, positions_rad: np.ndarray) -> None:
        """
        Send a joint position command (radians) to the arm.

        The command is clipped to the configured joint limits, converted to
        degrees, and submitted as a single clear_set / set_joint_cmd_pose /
        send_cmd transaction. It is a no-op when not connected.
        """
        positions_rad = np.asarray(positions_rad, dtype=np.float32)
        if positions_rad.shape != (self.n_joints,):
            raise ValueError(f"Expected positions shape ({self.n_joints},), got {positions_rad.shape}")
        if not self._connected or self._robot is None:
            return
        clipped = np.clip(positions_rad, self.joint_limit_low, self.joint_limit_high)
        joints_deg = np.rad2deg(clipped).astype(np.float64).tolist()
        self._robot.clear_set()
        self._robot.set_joint_cmd_pose(self.arm, joints_deg)
        self._robot.send_cmd()

    def get_arm_state(self) -> Tuple[int, int]:
        """Return (cur_state, err_code) for this arm."""
        if not self._connected or self._robot is None or self._dcss is None:
            return 0, 0
        data = self._robot.subscribe(self._dcss)
        states = data["states"][self.arm_index]
        return int(states["cur_state"]), int(states["err_code"])

    def clear_errors(self) -> None:
        """Clear errors on this arm. No-op when not connected."""
        if not self._connected or self._robot is None:
            return
        self._robot.clear_set()
        self._robot.clear_error(self.arm)
        self._robot.send_cmd()

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable (position mode) or disable (idle) the arm.

        enabled=True -> set_state(ARM_STATE_POSITION); enabled=False -> set_state(ARM_STATE_IDLE).
        No-op when not connected.
        """
        if not self._connected or self._robot is None:
            return
        state = self.ARM_STATE_POSITION if enabled else self.ARM_STATE_IDLE
        self._robot.clear_set()
        self._robot.set_state(self.arm, state)
        self._robot.send_cmd()

    # ========== Kinematics ==========
    def _init_kine(self, kine_config_path: str) -> None:
        """Lazily initialize the Marvin_Kine forward-kinematics helper."""
        if kine_config_path == "default":
            kine_config_path = os.path.join(
                os.path.dirname(__file__), "sdk", "config", "ccs_m3.MvKDCfg"
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
