"""
Preview /home/tianji/Downloads/demo2_shoulder.csv in a MuJoCo passive viewer.

Runs the same mink IK precompute as test_run_csv_mink.py but DOES NOT connect
to the real robot. Instead it animates the precomputed joint trajectory in
scene_tianji_wuji_full.xml via mujoco.viewer.launch_passive, so you can
sanity-check the motion before running the robot script.

Usage:
    python3 test_view_csv_mink.py                 # play at real-time 60 Hz
    python3 test_view_csv_mink.py 0.25            # quarter speed (very slow)
    python3 test_view_csv_mink.py 1.0 --loop      # loop forever

Controls (from mujoco.viewer):
    - Close the window to stop playback.
    - Drag / scroll to orbit the camera.
"""

import sys
import os
import time
import logging

import numpy as np

DEXTELE_DIR = '/home/tianji/Desktop/tianji_teleop/DexTele'
sys.path.insert(0, DEXTELE_DIR)

# Load the shared helpers (load_trajectory, build_sim_and_ik, precompute, etc.)
# from test_run_csv_mink.py without running its main body.
_RUNNER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'test_run_csv_mink.py',
)
with open(_RUNNER_PATH) as _f:
    _src = _f.read()
_marker = 'speed_factor = float(sys.argv[1])'
if _marker not in _src:
    raise RuntimeError(f"could not find marker '{_marker}' in {_RUNNER_PATH}")
_runner_ns = {'__name__': 'test_run_csv_mink', '__file__': _RUNNER_PATH}
exec(_src[:_src.index(_marker)], _runner_ns)

CSV_PATH = _runner_ns['CSV_PATH']
load_trajectory = _runner_ns['load_trajectory']
build_sim_and_ik = _runner_ns['build_sim_and_ik']
precompute = _runner_ns['precompute']

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('mink_viewer')
logger.setLevel(logging.INFO)

FRAME_HZ = 60
FRAME_PERIOD_S = 1.0 / FRAME_HZ

# --- args ---
_loop = False
_positional = []
for arg in sys.argv[1:]:
    if arg in ('--loop', '-l'):
        _loop = True
    else:
        _positional.append(arg)
speed_factor = float(_positional[0]) if _positional else 1.0
if speed_factor <= 0:
    raise ValueError(f"speed_factor must be > 0 (got {speed_factor})")

logger.info(f"Loading trajectory: {CSV_PATH}")
frames = load_trajectory(CSV_PATH)
duration = frames[-1][0] - frames[0][0]
rate = (len(frames) - 1) / duration if duration > 0 else 0.0
logger.info(f"  {len(frames)} frames, duration {duration:.2f}s, rate {rate:.1f} Hz")
logger.info(f"  speed factor: {speed_factor}x "
            f"(playback duration {duration / speed_factor:.2f}s per pass)")

logger.info("Building sim + mink IK ...")
model, data, ik = build_sim_and_ik()

logger.info("Pre-computing joint trajectory (mink) ...")
joint_traj = precompute(frames, model, data, ik)

# joint_traj entries are (t, ja_deg_list, jb_deg_list).
# Pre-compute the qpos arrays in radians for fast replay.
_ja_rad = np.deg2rad(np.array([jt[1] for jt in joint_traj]))  # (N, 7)
_jb_rad = np.deg2rad(np.array([jt[2] for jt in joint_traj]))  # (N, 7)
_t_csv = np.array([jt[0] for jt in joint_traj])
_n = len(joint_traj)
_left_addrs = list(ik.left.qpos_addrs)
_right_addrs = list(ik.right.qpos_addrs)

# Seed sim to frame 0 before opening the viewer.
for i, addr in enumerate(_left_addrs):
    data.qpos[addr] = _ja_rad[0, i]
for i, addr in enumerate(_right_addrs):
    data.qpos[addr] = _jb_rad[0, i]
mujoco.mj_forward(model, data)

logger.info(f"Opening viewer — {_n} frames, {'looping' if _loop else 'single pass'}")

def _play_once(viewer):
    """Step through the full trajectory once, pacing to the CSV's clock."""
    t_start = time.monotonic()
    t0_csv = _t_csv[0]
    k = 0
    while k < _n:
        if not viewer.is_running():
            return False
        target_t = t_start + (_t_csv[k] - t0_csv) / speed_factor
        now = time.monotonic()
        if target_t > now:
            time.sleep(target_t - now)

        # Find the frame that matches the current playback time (catch up if
        # we're behind so the motion stays on-time rather than compressing).
        elapsed = (time.monotonic() - t_start) * speed_factor
        while k + 1 < _n and (_t_csv[k + 1] - t0_csv) <= elapsed:
            k += 1

        for i, addr in enumerate(_left_addrs):
            data.qpos[addr] = _ja_rad[k, i]
        for i, addr in enumerate(_right_addrs):
            data.qpos[addr] = _jb_rad[k, i]
        mujoco.mj_forward(model, data)
        viewer.sync()
        k += 1
    return True


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.lookat[:] = [0.15, 0.0, 0.95]
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 180
    viewer.sync()

    while viewer.is_running():
        finished = _play_once(viewer)
        if not finished:
            break
        if not _loop:
            break
        # Small pause at the end of each pass before restarting
        for _ in range(int(0.5 * FRAME_HZ)):
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(FRAME_PERIOD_S)

    # Hold the final frame briefly so you can see the end pose
    for _ in range(int(1.0 * FRAME_HZ)):
        if not viewer.is_running():
            break
        viewer.sync()
        time.sleep(FRAME_PERIOD_S)

logger.info("Done.")
