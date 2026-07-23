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

3. **Kinematics config (optional).** A default `ccs_m3.MvKDCfg` config ships inside `unienv_tianji/sdk/config/`. Pass `kine_config_path="default"` to use it, or supply your own `.MvKDCfg` path.

## Usage

```python
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld
from unienv_tianji import TianjiArmActor
import numpy as np

world = RealWorld(
    NumpyComputeBackend,
    world_timestep=0.04,  # Usually set this to exactly the control timestep
    batch_size=None,  # None means single instance
)

actor = TianjiArmActor(
    world,
    ip="192.168.1.190",
    arm="A",  # "A" (left, index 0) or "B" (right, index 1)
    vel_ratio=10,
    acc_ratio=10,
    control_timestep=0.04,
    update_timestep=0.04,
    # kine_config_path="default",  # uncomment to add a "tcp_pose" (4x4) FK observation
)

rng = np.random.default_rng(42)
actor.reset()

while True:
    obs = actor.get_observation()
    print(obs)
    rng, action = actor.action_space.sample(rng)
    actor.set_next_action(action)
    actor.pre_environment_step(0.04)
    actor.post_environment_step(0.04)
```

### Testing without hardware

```python
actor = TianjiArmActor(
    world,
    arm="A",
    connect=False,  # skip all hardware I/O; reads return zeros, sends are no-ops
)
```
## License

This repository is MIT licensed (see [LICENSE](LICENSE)).

The files under `unienv_tianji/sdk/` (`fx_robot.py`, `fx_kine.py`, and the
`libMarvinSDK.*` / `libKine.*` prebuilt binaries) are vendored from the official
[TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK)
repository and remain Apache-2.0 licensed, Copyright 2025 上海孚晞科技有限公司 —
see [`unienv_tianji/sdk/LICENSE`](unienv_tianji/sdk/LICENSE) and the per-file
provenance headers.
