# Tianji / Marvin Robot Arm Adaptor

Based on the official [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK) `SDK_PYTHON` (Apache-2.0, Copyright 2025 上海孚晞科技有限公司).

## Installation

```bash
pip install unienv-tianji
```

> **Note.** End-effector-pose control (the :class:`TianjiArmEefActor` node below)
> depends on [`mujoco`](https://github.com/google-deepmind/mujoco) and
> [`mink`](https://github.com/kevinmks/mink) (with the `osqp` QP solver) for
> inverse kinematics. These are listed as dependencies and installed
> automatically. Pure joint-position control via :class:`TianjiArmActor` does
> not require them at runtime.

## Hardware setup

The Tianji/Marvin dual-arm robot is controlled over Ethernet through a dedicated controller cabinet.

1. **Network.** Connect the controller to your host machine (directly or via a switch) and configure your host's IPv4 address to be on the same subnet as the controller, e.g. `192.168.1.x`. Verify reachability with `ping 192.168.1.190` (the default controller IP).

2. **Native SDK binaries (NOT included).** The Python SDK in `unienv_tianji/sdk/SDK_PYTHON/` (`fx_robot.py`, `fx_kine.py`) loads native shared libraries via `ctypes`, resolving them **relative to the vendored file location** (i.e. inside `unienv_tianji/sdk/SDK_PYTHON/`):
   - `libMarvinSDK.so` (Linux) / `libMarvinSDK.dll` (Windows) — required for any robot communication (the `Marvin_Robot`/`DCSS` link).
   - `libKine.so` (Linux) / `libKine.dll` (Windows) — only required for the **legacy** vendor kinematics path (`kine_config_path`, `TianjiArmActor.send_eef_command`/`read_tcp_pose`, and the optional `tcp_pose` observation).

   These vendor binaries are **NOT shipped with this package** (they are
   git-ignored build artifacts). You can either download prebuilt ones from the
   official [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK)
   repository (`SDK_PYTHON/`), or **build them from the vendored C++ sources** —
   each SDK directory ships a `makefile` for Linux (and `makefile_dll` /
   upstream `marvinSDK_ubuntu.sh` / `marvinSDK_windows.bat` for Windows):

   ```bash
   # libMarvinSDK.so (control/communication library)
   cd unienv_tianji/sdk/contrlSDK && make

   # libKine.so (legacy vendor IK/FK library)
   cd ../kinematicsSDK && make

   # install next to the ctypes bindings
   cp ../contrlSDK/libMarvinSDK.so libKine.so ../SDK_PYTHON/
   ```

   The makefiles only need `g++` and standard Linux libs (`-lpthread -lrt`):
   they run `g++ *.cpp -Wall -w -O2 -fPIC -shared ...` (with `-DCMPL_LIN` for
   `libMarvinSDK.so`). The resulting files must end up at:

   ```
   <site-packages>/unienv_tianji/sdk/SDK_PYTHON/libMarvinSDK.so
   <site-packages>/unienv_tianji/sdk/SDK_PYTHON/libKine.so
   ```

   **Offline mode (`connect=False`) and the mink-based `TianjiArmEefActor` IK work without any native binaries** — `TianjiArmEefActor` resolves IK against a self-contained kinematics-only MJCF (shipped under `unienv_tianji/assets/`), so it does not need `libKine`; `libKine` is only needed for the legacy `send_eef_command`/`read_tcp_pose` path on `TianjiArmActor`. Without `libMarvinSDK.so` present, instantiate the actor with `connect=False` (see below) — all hardware reads return zeros and command sends become no-ops, which is useful for testing the integration without hardware.

3. **Kinematics config (optional).** A default `ccs_m6_40.MvKDCfg` config (for the M6-S-R-CCS-696) ships inside `unienv_tianji/sdk/`. Pass `kine_config_path="default"` to use it, or supply your own `.MvKDCfg` path.

## Usage

> **One connection per process.** The vendored SDK binds a fixed UDP port and
> exchanges data through a single shared-memory structure, so **only one
> controller connection can exist per process**. To drive both arms (``"A"``
> and ``"B"``) from the same process, create a single
> :class:`TianjiConnection` and pass it to both actors via ``connection=``.
> ``connect=False`` (offline/test mode) needs no connection.

```python
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld
from unienv_tianji import TianjiArmActor, TianjiConnection
import numpy as np

world = RealWorld(
    NumpyComputeBackend,
    world_timestep=0.04,  # Usually set this to exactly the control timestep
    batch_size=None,  # None means single instance
)

# One shared connection for the whole process (dual-arm safe).
connection = TianjiConnection("192.168.1.190")

actor_a = TianjiArmActor(
    world,
    ip="192.168.1.190",
    arm="A",  # "A" (left, index 0) or "B" (right, index 1)
    vel_ratio=10,
    acc_ratio=10,
    control_timestep=0.04,
    update_timestep=0.04,
    connection=connection,
    # kine_config_path="default",  # uncomment to add a "tcp_pose" (4x4) FK observation
)

# A second actor sharing the same connection:
actor_b = TianjiArmActor(
    world,
    arm="B",
    vel_ratio=10,
    acc_ratio=10,
    connection=connection,
)

rng = np.random.default_rng(42)
actor_a.reset()
actor_b.reset()

try:
    while True:
        obs_a = actor_a.get_observation()
        obs_b = actor_b.get_observation()
        rng, action_a = actor_a.action_space.sample(rng)
        rng, action_b = actor_b.action_space.sample(rng)
        actor_a.set_next_action(action_a)
        actor_b.set_next_action(action_b)
        actor_a.pre_environment_step(0.04)
        actor_b.pre_environment_step(0.04)
        actor_a.post_environment_step(0.04)
        actor_b.post_environment_step(0.04)
finally:
    # Disable each arm, then release the shared connection last.
    actor_a.close()
    actor_b.close()
    connection.close()
```

### Testing without hardware

```python
# Offline mode: no hardware I/O. Reads return zeros, sends are no-ops.
# No TianjiConnection is required.
actor = TianjiArmActor(
    world,
    arm="A",
    connect=False,
)
```

### Observations & hardware errors

`TianjiArmActor` exposes its full proprioception — every channel that rides in
the SDK's regular 1kHz feedback frame (`RT_OUT`, returned by `subscribe()`) —
as observation keys. Error/state codes do **not** become observations; they raise
instead (see below).

| key                               | shape  | unit            | source (`RT_OUT` field)             |
|-----------------------------------|--------|-----------------|-------------------------------------|
| `joint_positions`                 | (7,)   | rad             | `m_FB_Joint_Pos` (deg → rad)        |
| `joint_velocities`                | (7,)   | rad/s           | `m_FB_Joint_Vel` (deg/s → rad/s)    |
| `joint_torques`                   | (7,)   | N·m             | `m_FB_Joint_SToq` (sensor torque)    |
| `joint_currents`                  | (7,)   | per-mille (0/00) | `m_FB_Joint_CToq` (motor current, ‰ of rated) |
| `joint_temperatures`              | (7,)   | °C              | `m_FB_Joint_Them`                    |
| `joint_friction_estimates`        | (7,)   | N·m             | `m_EST_Joint_Firc`                   |
| `joint_external_force_estimates`  | (7,)   | N·m             | `m_EST_Joint_Force`                  |
| `cartesian_force_estimate`        | (6,)   | N / N·m         | `m_EST_Cart_FN` (6D end-effector force estimate) |
| `tcp_pose` (optional, `kine_config_path`) | (4,4) | homogeneous matrix | FK of current joints in the SDK base frame (mm) |

All extended channels are read from the **same** feedback frame as the basic
trio — there are **no extra blocking/vendor calls per step**. Set
`read_extended_proprioception=False` on the constructor to expose only the
basic `joint_positions` / `joint_velocities` / `joint_torques` trio (matching
the pre-extension API).

> **Motor-side torque.** The SDK provides a motor current channel
> (`m_FB_Joint_CToq`, exposed as `joint_currents` in per-mille of the rated
> current), but it does **not** expose a direct motor-side torque sensor —
> `joint_torques` (`m_FB_Joint_SToq`) is the joint-side / link-side sensor
> torque. There is no genuine motor-torque sensor channel available.

The previous `arm_state` and `error_code` observation keys have been **removed**
— they are not policy-relevant. Instead, the actor reads a hardware fault from
the feedback channel at every refresh (``post_environment_step`` /
``after_reset`` / ``after_reload``): if the arm reports a non-zero ``err_code``
**or** ``cur_state == 100`` (``ARM_STATE_ERROR``), it raises
:class:`TianjiArmHardwareError` (a ``RuntimeError`` subclass carrying the arm
id, ``cur_state`` and ``err_code``). State 0/idle is **not** an error.

Recover with :meth:`TianjiArmActor.clear_errors`, then re-enable the arm:

```python
from unienv_tianji import TianjiArmHardwareError

try:
    env.step(action)
except TianjiArmHardwareError as e:
    print(f"arm {e.arm} fault: cur_state={e.cur_state} err_code={e.error_code}")
    actor.clear_errors()
    actor.set_enabled(True)   # re-runs the full enable sequence
```

`TianjiArmEefActor` passes the child's extended-proprioception channels through
in its observation dict alongside its own six keys
(`joint_position`/`joint_velocity`/`joint_torque`/`eef_position`/`eef_quaternion`/`last_ik_solved_error`);
the passthrough keys keep the child's plural adaptor names (e.g.
`joint_currents`, `joint_temperatures`, `cartesian_force_estimate`).

## End-effector pose control — `TianjiArmEefActor`

`TianjiArmEefActor` wraps one :class:`TianjiArmActor` and exposes an absolute
end-effector-pose action space matching the sim's ``EndEffectorPoseController``
contract: a 3D position + a rotation in one of
``{"euler"`` (intrinsic XYZ, ``R = Rx @ Ry @ Rz``), ``"quat"`` (wxyz),
``"rot6d"``}``. IK is solved with [`mink`](https://github.com/kevinmks/mink)
against a kinematics-only per-arm MJCF, seeded each step from the child
actor's *current* feedback joint positions (solution continuity, like the sim's
DLS IK). The resulting joint targets are sent through the child actor
(``send_joint_command``), so it works in every control mode.

EEF actions and observations are always expressed directly in the robot base
frame, which is the MJCF robot-root frame. The sim anchors the robot at the
world origin, so its robot base and world frames coincide. Teleoperation
retargeting nodes are responsible for transforming poses from controller,
camera, task, or other source frames into the robot base frame.

> **Rotation conversions.** Action rotation parsing (euler/quat/rot6d) and the
> EEF quaternion observation call [`xbarray`](https://github.com/UniEnvOrg/XBArray)'s
> numpy-backend rotation conversions directly
> (``euler_angles_to_matrix``, ``matrix_to_euler_angles``,
> ``quaternion_to_matrix``, ``matrix_to_quaternion``,
> ``rotation_6d_to_matrix``, ``matrix_to_rotation_6d``) — the **single source
> of truth for the rotation math** shared with the sim stack, so the real/spec
> and sim cannot drift. Requires XBArray ≥ 0.0.1a19 (the numpy backend's
> ``matrix_to_euler_angles`` must work on raw numpy arrays; the fix for this
> lands in the a19 release — until then it lives in the XBArray working tree,
> replacing the broken a18 path that invoked the array-api callable ``.size()``
> on plain numpy). No local rotation helpers remain.

### Quickstart — hold pose

```python
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld, WorldEnv
from unienv_tianji import TianjiConnection, TianjiArmEefActor
from xbarray.transformations.rotation_conversions.numpy import (
    quaternion_to_matrix, matrix_to_euler_angles,
)
import numpy as np

world = RealWorld(NumpyComputeBackend, world_timestep=0.04, batch_size=None)
connection = TianjiConnection("192.168.1.190")

actor = TianjiArmEefActor(
    world,
    arm="A",                       # "A" = left, "B" = right
    connection=connection,
    connect=True,
    rotation_representation="euler",  # or "quat" / "rot6d"
    normalize_rotation_action=False,
    control_mode="cart_impedance",  # forwarded to the child actor
)
env = WorldEnv(world, actor)

_, obs, _ = env.reset()
eef_pos = obs["eef_position"]                 # (3,) metres, robot base frame
eef_quat = obs["eef_quaternion"]              # (4,) wxyz, robot base frame
# Build a "hold current pose" euler action from the obs:
eul = matrix_to_euler_angles(quaternion_to_matrix(eef_quat.astype(np.float64)), "XYZ")
action = np.concatenate([eef_pos, eul]).astype(np.float32)
env.step(action)
```

Action / observation spaces:

| mode / repr            | action shape | obs keys                                                |
|------------------------|--------------|---------------------------------------------------------|
| `eef_pose` + `euler`   | (6,)         | `joint_position(7)`, `joint_velocity(7)`, `joint_torque(7)`, `eef_position(3)`, `eef_quaternion(4 wxyz)`, `last_ik_solved_error(6)` |
| `eef_pose` + `quat`    | (7,)         | (same as above)                                         |
| `eef_pose` + `rot6d`   | (9,)         | (same as above)                                         |
| `joint_position`       | (7,)         | (same as above; `last_ik_solved_error` stays zeros)    |

### Velocity limits

`TianjiArmEefActor` defaults to a node-side `max_joint_pos_vel=0.2` rad/s,
matching the sim's `tianji_marvin_wuji` limit. When it constructs its internal
`TianjiArmActor`, the vendor `vel_ratio` and `acc_ratio` default to `10`
(about 18 deg/s or 0.31 rad/s), leaving headroom so the node-side clip is the
binding motion contract and the firmware cap only guards against abuse.

Override both layers through the EEF actor constructor, for example:

```python
actor = TianjiArmEefActor(
    world,
    arm="A",
    connection=connection,
    max_joint_pos_vel=0.15,  # rad/s; use None to disable the node-side clip
    vel_ratio=12,            # forwarded to the internally constructed child
    acc_ratio=12,
)
```

If supplying `arm_actor=`, configure its `vel_ratio` / `acc_ratio` when creating
that child actor instead.

`last_ik_solved_error = [pos_err(3), rot_err(3)]` from the last IK solve (zeros
before the first eef action; `rot_err` is the axis-angle vector of the
orientation error). The EEF pose is the per-arm MJCF palm-frame FK reported
directly in the robot base / MJCF robot-root frame, with no configurable frame
transform. Hardware faults propagate as :class:`TianjiArmHardwareError` (same
as the child actor).

### EEF coordinate-frame convention

Absolute EEF actions and EEF pose observations use the robot base frame
(identical to the MJCF robot-root frame) for both arms. In simulation the robot
is anchored at the world origin, so this frame also coincides with the sim world
frame. Any source-frame calibration or transformation—for example from a
teleoperation controller, tracker, camera, or task frame—belongs in the teleop
retargeting node before actions reach `TianjiArmEefActor`.

### MJCF provenance

The kinematics-only per-arm MJCFs ship under `unienv_tianji/assets/`
(`tianji_marvin_left_arm_kine.xml`, `tianji_marvin_right_arm_kine.xml`). They
are derived from the full dual-arm + Wuji-hand source MJCF in
`genesis_adaptor/unienv_genesis_collection/.../tianji_marvin_CCS_real_limit_wuji.xml`
by `scripts/generate_kine_mjcf.py`: each file keeps one 7-DoF arm chain (joints,
inertials, defaults, ranges), drops the other arm, all meshes/geoms/materials,
finger joints, actuators, sensors and contacts, welds the palm body to Link7,
adds a `palm` site replicating the palm body frame (the IK target frame), and
adds a `home` keyframe. Re-run `python scripts/generate_kine_mjcf.py` to
regenerate them if the source MJCF changes.

## Control modes

`TianjiArmActor` supports three control modes via the `control_mode` kwarg. In
every mode the motion target is a **joint position command**
(`set_joint_cmd_pose`); the difference is *how compliantly* the arm tracks it.

| `control_mode`     | SDK state | Impedance type | What it does                                                                 |
|--------------------|-----------|----------------|------------------------------------------------------------------------------|
| `"position"`       | 1 (position-follow) | —        | Rigid joint tracking. The arm holds the commanded pose as tightly as it can. |
| `"joint_impedance"`| 3 (torque)         | 1 (joint) | Compliant joint tracking: each joint behaves like a spring-damper around the commanded joint target. Good for safe contact / hand-guided tweaks. |
| `"cart_impedance"` | 3 (torque)         | 2 (cartesian) | Compliant Cartesian tracking: the end-effector behaves like a Cartesian spring-damper around the commanded TCP pose (derived from the joint targets). Good for force / push-along-surface tasks. |

> **Impedance = compliant tracking of the same joint targets.** You still call
> `set_next_action` / `send_joint_command` with joint positions; the controller
> tracks them with the configured stiffness/damping instead of rigidly.

### Stiffness / damping (KD) defaults and units

- **`joint_kd=(K, D)`** (used by `joint_impedance`):
  - `K`: per-joint stiffness in **N·m/deg**, recommended ≤ 2 per joint.
    Default `K = [2, 2, 2, 1, 1, 1, 1]` (stiffer on the base joints, softer on the wrist).
  - `D`: per-joint damping in **N·m/(deg/s)**, range 0–1.
    Default `D = [0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2]`.
- **`cart_kd=(K, D)`** (used by `cart_impedance`):
  - `K[0:3]` translational stiffness in **N/m** (≤ 3000); `K[3:6]` rotational
    stiffness in **N·m/rad** (≤ 100); `K[6]` null-space stiffness (≤ 20).
    Default `K = [3000, 3000, 3000, 60, 60, 60, 0]`.
  - `D`: damping *ratios*, 0–1 (per axis + null-space).
    Default `D = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0]`.

### Tool payload

Impedance modes apply the controller's gravity/compensation using a tool
payload model. Pass `set_tool_payload=True` (default) so the enable sequence
calls `set_tool` with:

- `tool_mass` (kg), default `1.40` — tuned on arm A hardware (wuji hand +
  mount + cabling) via joint-impedance droop measurements. Re-tune if the
  mounted payload changes.
- `tool_com` (mm, flange frame), default `(0.0, 0.0, 80.0)` — center of mass
  80 mm out along the flange Z axis. **Units are millimeters** (per the
  `fx_robot.set_tool` docstring).
- `tool_inertia` `(Ixx, Ixy, Ixz, Iyy, Iyz, Izz)`, default all zero.

Set `set_tool_payload=False` to skip the `set_tool` call entirely.

### End-effector (TCP) position control

The SDK has **no streaming cartesian command**; EEF control is implemented as
IK → joint targets. Enable kinematics with `kine_config_path="default"` and use
`send_eef_command(pose_4x4)`, which seeds IK with the current joint
configuration (to avoid branch jumps), raises `RuntimeError` on IK failure
(unreachable / singular — no command is sent in that case), and otherwise
dispatches through `send_joint_command`. This works in every control mode.

```python
connection = TianjiConnection("192.168.1.190")
actor = TianjiArmActor(
    world,
    arm="A",
    connection=connection,
    control_mode="cart_impedance",
    kine_config_path="default",  # required for send_eef_command / tcp_pose obs
)

# Command a 4x4 homogeneous TCP pose, expressed in the arm's SDK base frame
# (x forward, y left, z up; translations in MILLIMETERS).
target_pose = np.eye(4, dtype=np.float64)
target_pose[0, 3] = 300.0  # x = 300 mm forward
target_pose[2, 3] = 450.0  # z = 450 mm up
actor.send_eef_command(target_pose)

# Read the current TCP pose (forward kinematics of current joints).
current_pose = actor.read_tcp_pose()  # (4, 4) ndarray
```

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).

The files under `unienv_tianji/sdk/` (`fx_robot.py`, `fx_kine.py`, and the
`libMarvinSDK.*` / `libKine.*` prebuilt binaries) are vendored from the official
[TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK)
repository and remain Apache-2.0 licensed, Copyright 2025 上海孚晞科技有限公司 —
see [`unienv_tianji/sdk/LICENSE`](unienv_tianji/sdk/LICENSE) and the per-file
provenance headers.
