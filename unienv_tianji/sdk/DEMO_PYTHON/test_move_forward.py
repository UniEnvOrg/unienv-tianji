"""
Move both arms forward (+X in WORLD frame) from their current pose.

Reads the robot's current joint positions, mirrors them into the MuJoCo scene
(scene_tianji_wuji_full.xml from the calvinzqiu DexTele fork), computes each
arm's current EE pose from the MuJoCo site, then uses mink to solve IK for a
straight-line world-frame motion that advances each EE by FORWARD_M along
world +X while keeping orientation fixed. Streams the resulting joint
trajectory to both arms at 1 kHz in position-follow mode.

The Tianji SDK's kinematics operates in each arm's DH-local frame, which is
NOT aligned with world +X — the zero-joint FK returns (0, 0, 870) (straight
along local +Z). Mink sidesteps that entire problem by working in the URDF's
world frame directly.

Usage:
    python3 test_move_forward.py              # 100 mm forward, 5 s duration
    python3 test_move_forward.py 0.05         # 50 mm forward
    python3 test_move_forward.py 0.1 8        # 100 mm forward over 8 s

CAUTION: check both arms have the requested clearance ahead of their TCPs
before running.
"""

import sys
import os
import time
import logging

import numpy as np

# DexTele + SDK import setup (same as test_run_csv_mink.py)
DEXTELE_DIR = '/home/tianji/Desktop/tianji_teleop/DexTele'
SCENE_XML = os.path.join(DEXTELE_DIR, 'robots/tianji/scene_tianji_wuji_full_fixed.xml')
sys.path.insert(0, DEXTELE_DIR)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_tianji_arm_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _tianji_arm_dir)

import mujoco  # noqa: E402
from teleop.tianji_arm_ik_mink import TianjiDualArmIKMink  # noqa: E402
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS  # noqa: E402

ROBOT_IP = '192.168.1.190'
DEFAULT_FORWARD_M = 0.1     # 10 cm
DEFAULT_DURATION_S = 5.0    # slow
IK_DT = 1.0 / 60.0
IK_MAX_ITERS = 5
STREAM_FREQ_HZ = 1000
STREAM_PERIOD_S = 1.0 / STREAM_FREQ_HZ

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('forward')
logger.setLevel(logging.INFO)

forward_m  = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FORWARD_M
duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DURATION_S
if duration_s <= 0:
    raise ValueError("duration must be > 0")


def fmt(vals, decimals=2):
    return "[" + ", ".join(f"{v:+.{decimals}f}" for v in vals) + "]"


def site_quat_wxyz(data, site_id):
    mat = data.site_xmat[site_id].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


# ---------- Sim + mink ----------
logger.info(f"Loading scene: {SCENE_XML}")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
ik = TianjiDualArmIKMink(model, data)
if ik.left is None or ik.right is None:
    raise RuntimeError("scene is missing left or right arm bodies")

# ---------- Robot connect ----------
dcss = DCSS()
robot = Marvin_Robot()
logger.info(f"Connecting to robot at {ROBOT_IP} ...")
if robot.connect(ROBOT_IP) == 0:
    logger.error("Connection failed. Exiting.")
    sys.exit(1)
time.sleep(0.5)

frame_prev = None
ok = False
for _ in range(10):
    sub = robot.subscribe(dcss)
    fs = sub['outputs'][0]['frame_serial']
    if fs != 0 and fs != frame_prev:
        ok = True
        frame_prev = fs
    time.sleep(0.1)
if not ok:
    logger.error("No data from robot. Exiting.")
    robot.release_robot()
    sys.exit(1)
logger.info("Robot connected.")

robot.clear_set()
robot.clear_error('A')
robot.clear_error('B')
robot.send_cmd()
time.sleep(0.5)

# Read current joints (degrees in SDK convention)
sub = robot.subscribe(dcss)
cur_a_deg = np.array(sub["outputs"][0]["fb_joint_pos"][:7], dtype=np.float64)
cur_b_deg = np.array(sub["outputs"][1]["fb_joint_pos"][:7], dtype=np.float64)
logger.info(f"  current joints A (left):  {fmt(cur_a_deg)}")
logger.info(f"  current joints B (right): {fmt(cur_b_deg)}")

# Mirror robot joints into sim (mink expects radians in data.qpos)
for i, addr in enumerate(ik.left.qpos_addrs):
    data.qpos[addr] = np.deg2rad(cur_a_deg[i])
for i, addr in enumerate(ik.right.qpos_addrs):
    data.qpos[addr] = np.deg2rad(cur_b_deg[i])
# Posture target stays at current so the null-space bias doesn't drag joints
ik.left.set_posture_target(np.deg2rad(cur_a_deg))
ik.right.set_posture_target(np.deg2rad(cur_b_deg))
mujoco.mj_forward(model, data)

# Current EE poses in world frame
start_pos_l = data.site_xpos[ik.left.ee_site_id].copy()
start_pos_r = data.site_xpos[ik.right.ee_site_id].copy()
start_quat_l = site_quat_wxyz(data, ik.left.ee_site_id)
start_quat_r = site_quat_wxyz(data, ik.right.ee_site_id)
logger.info(f"  A start EE world: {fmt(start_pos_l, decimals=3)} m")
logger.info(f"  B start EE world: {fmt(start_pos_r, decimals=3)} m")

end_pos_l = start_pos_l + np.array([forward_m, 0.0, 0.0])
end_pos_r = start_pos_r + np.array([forward_m, 0.0, 0.0])
logger.info(f"  target delta: +{forward_m*1000:.1f} mm along world +X")

# ---------- Enter position mode at low vel for safety ----------
robot.clear_set()
robot.set_state(arm='A', state=1)
robot.set_vel_acc(arm='A', velRatio=5, AccRatio=5)
robot.set_state(arm='B', state=1)
robot.set_vel_acc(arm='B', velRatio=5, AccRatio=5)
robot.send_cmd()
time.sleep(1.0)

# ---------- Precompute joint trajectory via mink ----------
n_ik_frames = max(int(duration_s * 60), 2)  # 60 Hz IK
logger.info(f"Pre-computing {n_ik_frames} IK frames over {duration_s:.1f}s ...")

joint_traj_a = []  # list of deg-lists
joint_traj_b = []
max_l_err = 0.0
max_r_err = 0.0

with np.errstate(all='ignore'):
    for k in range(n_ik_frames):
        alpha = (k + 1) / n_ik_frames
        tgt_l = start_pos_l + alpha * (end_pos_l - start_pos_l)
        tgt_r = start_pos_r + alpha * (end_pos_r - start_pos_r)

        for solver, tgt_pos, tgt_quat in (
            (ik.left, tgt_l, start_quat_l),
            (ik.right, tgt_r, start_quat_r),
        ):
            solver.sync_configuration(data)
            q_ik = solver.solve(tgt_pos, target_quat=tgt_quat,
                                dt=IK_DT, max_iters=IK_MAX_ITERS)
            if not np.all(np.isfinite(q_ik)):
                logger.error(f"frame {k}: IK returned non-finite joints, aborting")
                robot.clear_set()
                robot.set_state(arm='A', state=0)
                robot.set_state(arm='B', state=0)
                robot.send_cmd()
                robot.release_robot()
                sys.exit(1)
            for i, addr in enumerate(solver.qpos_addrs):
                data.qpos[addr] = q_ik[i]
        mujoco.mj_forward(model, data)

        l_err = float(np.linalg.norm(data.site_xpos[ik.left.ee_site_id] - tgt_l))
        r_err = float(np.linalg.norm(data.site_xpos[ik.right.ee_site_id] - tgt_r))
        max_l_err = max(max_l_err, l_err)
        max_r_err = max(max_r_err, r_err)

        qa = np.rad2deg([data.qpos[a] for a in ik.left.qpos_addrs]).tolist()
        qb = np.rad2deg([data.qpos[a] for a in ik.right.qpos_addrs]).tolist()
        joint_traj_a.append(qa)
        joint_traj_b.append(qb)

logger.info(f"  max EE tracking err: left={max_l_err*1000:.1f}mm right={max_r_err*1000:.1f}mm")
logger.info(f"  first joints A: {fmt(joint_traj_a[0])}")
logger.info(f"  first joints B: {fmt(joint_traj_b[0])}")
logger.info(f"  last  joints A: {fmt(joint_traj_a[-1])}")
logger.info(f"  last  joints B: {fmt(joint_traj_b[-1])}")

# Sanity check: the precompute's "first joints" should be near the real robot
# joints (since the first IK frame is alpha=1/n_ik_frames ≈ near start). If
# they're wildly different, mink found a very different branch — bail.
first_jump_a = max(abs(joint_traj_a[0][i] - cur_a_deg[i]) for i in range(7))
first_jump_b = max(abs(joint_traj_b[0][i] - cur_b_deg[i]) for i in range(7))
logger.info(f"  first-frame joint jump: A={first_jump_a:.2f} deg, B={first_jump_b:.2f} deg")
if max(first_jump_a, first_jump_b) > 15.0:
    logger.error("first IK frame jumps >15 deg from current robot joints. "
                 "Mink picked a very different branch. Aborting for safety.")
    robot.clear_set()
    robot.set_state(arm='A', state=0)
    robot.set_state(arm='B', state=0)
    robot.send_cmd()
    robot.release_robot()
    sys.exit(1)

# ---------- Upsample 60 Hz -> 1 kHz ----------
t_ik = np.linspace(0, duration_s, n_ik_frames)
n_stream = max(int(duration_s * STREAM_FREQ_HZ), 2)
t_stream = np.linspace(0, duration_s, n_stream)
arr_a = np.array(joint_traj_a)  # (n_ik_frames, 7)
arr_b = np.array(joint_traj_b)
stream_a = np.column_stack([np.interp(t_stream, t_ik, arr_a[:, j]) for j in range(7)])
stream_b = np.column_stack([np.interp(t_stream, t_ik, arr_b[:, j]) for j in range(7)])
logger.info(f"  upsampled to {n_stream} frames at {STREAM_FREQ_HZ} Hz")

# ---------- Streaming ----------
robot.clear_set()
robot.set_vel_acc(arm='A', velRatio=50, AccRatio=50)
robot.set_vel_acc(arm='B', velRatio=50, AccRatio=50)
robot.send_cmd()
time.sleep(0.3)

logger.info(f"Streaming {n_stream} frames at {STREAM_FREQ_HZ} Hz ...")
t_start = time.monotonic()
late = 0
for k in range(n_stream):
    target_t = t_start + k * STREAM_PERIOD_S
    now = time.monotonic()
    if target_t > now:
        time.sleep(target_t - now)
    else:
        late += 1
    robot.clear_set()
    robot.set_joint_cmd_pose(arm='A', joints=stream_a[k].tolist())
    robot.set_joint_cmd_pose(arm='B', joints=stream_b[k].tolist())
    robot.send_cmd()
elapsed = time.monotonic() - t_start
logger.info(f"  streaming done in {elapsed:.2f}s (late {late}/{n_stream})")

time.sleep(0.5)

# ---------- Cleanup ----------
logger.info("Disabling servos and releasing robot ...")
robot.clear_set()
robot.set_state(arm='A', state=0)
robot.set_state(arm='B', state=0)
robot.send_cmd()
time.sleep(0.5)
robot.release_robot()
logger.info("Done.")
