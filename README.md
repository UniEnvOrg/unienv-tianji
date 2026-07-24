# Tianji / Marvin Robot Arm Adaptor

Based on the official [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK) `SDK_PYTHON` (Apache-2.0, Copyright 2025 上海孚晞科技有限公司).

## Installation

```bash
pip install unienv-tianji
```

## Hardware setup

The Tianji/Marvin dual-arm robot is controlled over Ethernet through a dedicated controller cabinet.

1. **Network.** Connect the controller to your host machine (directly or via a switch) and configure your host's IPv4 address to be on the same subnet as the controller, e.g. `192.168.1.x`. Verify reachability with `ping 192.168.1.190` (the default controller IP).

2. **Native SDK binaries.** The Python SDK in `unienv_tianji/sdk/` (`fx_robot.py`, `fx_kine.py`) loads native shared libraries via `ctypes`, resolving them **relative to the vendored file location** (i.e. inside `unienv_tianji/sdk/`):
   - `libMarvinSDK.so` (Linux) / `libMarvinSDK.dll` (Windows) — required for any robot communication.
   - `libKine.so` (Linux) / `libKine.dll` (Windows) — only required when forward kinematics (`kine_config_path`) is enabled.

   This package **ships the vendor's official prebuilt binaries** — Linux x86_64 `.so` and Windows `.dll` — taken directly from the official [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK) repository (`SDK_PYTHON/`), so on those platforms no extra build step is required. Users on other platforms can build the libraries from source using the upstream repo's `marvinSDK_ubuntu.sh` / `marvinSDK_windows.bat` scripts and place the resulting binaries next to the vendored files:

   ```
   <site-packages>/unienv_tianji/sdk/libMarvinSDK.so
   <site-packages>/unienv_tianji/sdk/libKine.so
   ```

   Without `libMarvinSDK.so` present, instantiate the actor with `connect=False` (see below) — all hardware reads return zeros and command sends become no-ops, which is useful for testing the integration without hardware.

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
