import sys
import os
import time
import math
import csv
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS
from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara

"""
Run a recorded dual-arm Cartesian trajectory CSV on the bi-arm CCS Marvin robot.

CSV format (header):
    waypoint, arm, sequence, x_m, y_m, z_m, qw, qx, qy, qz, time_s

Pipeline:
    1. Load CSV, group rows by timestamp into (left, right) frame pairs.
    2. Initialize kinematics for both arms (two Marvin_Kine instances).
    3. For each frame: build the world-frame TCP pose from (xyz, quat),
       transform into each arm's local frame using the URDF mount transforms,
       then run IK chaining ref joints frame-to-frame. Pick the least-violating
       branch from the SDK's 4 closed-form solutions. Abort if any frame can't
       fit within joint limits + LIMIT_TOL_DEG.
    4. Connect to robot, slow-move both arms to the first IK pose (position mode).
    5. Stream the joint commands at the CSV's native cadence (monotonic clock).
    6. Settle, disable servos, release.

Assumptions:
    - CSV positions are in METERS, expressed in the MuJoCo world frame
      defined by calvinzqiu/DexTele scene_tianji_wuji_full.xml.
    - Quaternions are (qw, qx, qy, qz), unit-length, Hamilton convention,
      expressing the TCP orientation in world frame.
    - TCP == flange (no tool offset). Add kine_*.set_tool_kine(...) if the
      data was recorded with a tool.
    - left -> Arm A (arm_type=0), right -> Arm B (arm_type=1).
"""

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('csv_runner')
logger.setLevel(logging.INFO)

DEFAULT_CSV_PATH = '/home/tianji/Downloads/demo2.csv'
DEFAULT_SPEED_FACTOR = 1.0  # 1.0=real-time, 0.5=half speed, 0.25=quarter speed
CONFIG_FILE = 'ccs_m6_40.MvKDCfg'
ROBOT_IP = '192.168.1.190'

# Joint-limit tolerance for IK branch selection. Solutions within this many
# degrees of the hard limit are accepted; the joints are clamped to the hard
# limit just before being sent to the robot. Real wrist overshoots in this
# data are typically only a few degrees.
LIMIT_TOL_DEG = 5.0

# CCS joint hard limits (deg). The MuJoCo URDF (calvinzqiu DexTele) reports
# joint1/3/5 mechanical range = +/-3.1103 rad = +/-178.2 deg, so we use 178
# here instead of the more conservative +/-170 in test_all_joints_position.py.
# These are still software-side checks; the firmware enforces its own limits.
JOINT_LO = [-178.0, -120.0, -178.0, -145.0, -178.0, -60.0, -90.0]
JOINT_HI = [ 178.0,  120.0,  178.0,   60.0,  178.0,  60.0,  90.0]

# Per-arm world->arm-local rigid transform, derived from
# calvinzqiu/DexTele robots/tianji/scene_tianji_wuji_full.xml:
#   <body name="left_base_link"  pos="0.0  0.04 1.01" quat="0.707105 -0.707108 0 0">
#   <body name="right_base_link" pos="0.0 -0.04 1.01" quat="0.707105  0.707108 0 0">
# T_world_armbase^-1 = [R^T, -R^T @ t; 0 1] in mm.
ARM_BASE_T_INV = {
    'left':  [
        [ 1.0,  0.0,  0.0,     0.00],
        [ 0.0,  0.0, -1.0,  1010.00],
        [ 0.0,  1.0,  0.0,   -40.00],
        [ 0.0,  0.0,  0.0,     1.00],
    ],
    'right': [
        [ 1.0,  0.0,  0.0,     0.00],
        [ 0.0,  0.0,  1.0, -1010.00],
        [ 0.0, -1.0,  0.0,   -40.00],
        [ 0.0,  0.0,  0.0,     1.00],
    ],
}

# Reference joints for the IK on the FIRST frame. Subsequent frames chain off
# the previous frame's IK solution to keep configuration continuous. This seed
# keeps J4 well away from the singular value 0.
FIRST_REF_JOINTS = [0.0, -60.0, 0.0, -90.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quat_pos_to_mat4x4(x_mm, y_mm, z_mm, qw, qx, qy, qz):
    """Position (mm) + unit quaternion (Hamilton, w first) -> 4x4 matrix."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise ValueError("zero-length quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),     x_mm],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),     y_mm],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy), z_mm],
        [0.0,               0.0,               0.0,               1.0],
    ]


def mat_mul_4x4(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j]
    return out


def world_to_arm_local(arm_label, x_m, y_m, z_m, qw, qx, qy, qz):
    """Build T_world_tcp from a CSV row, then return T_armlocal_tcp (mm)."""
    T_world_tcp = quat_pos_to_mat4x4(
        x_m * 1000.0, y_m * 1000.0, z_m * 1000.0, qw, qx, qy, qz
    )
    return mat_mul_4x4(ARM_BASE_T_INV[arm_label], T_world_tcp)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_trajectory(csv_path):
    """
    Return list of (time_s, left_local_4x4, right_local_4x4) sorted by time,
    with each TCP pose already transformed into its arm's local frame.
    """
    by_time = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = round(float(row['time_s']), 6)
            arm = row['arm'].strip().lower()
            if arm not in ('left', 'right'):
                raise ValueError(f"unknown arm '{arm}' in row: {row}")
            mat_local = world_to_arm_local(
                arm,
                float(row['x_m']), float(row['y_m']), float(row['z_m']),
                float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz']),
            )
            by_time.setdefault(t, {})[arm] = mat_local

    frames = []
    for t in sorted(by_time):
        entry = by_time[t]
        if 'left' not in entry or 'right' not in entry:
            raise ValueError(f"frame at t={t} missing left or right arm")
        frames.append((t, entry['left'], entry['right']))
    return frames


# ---------------------------------------------------------------------------
# Kinematics setup (two Marvin_Kine instances, one per arm)
# ---------------------------------------------------------------------------

def init_kinematics(config_path):
    kine_a = Marvin_Kine()
    kine_b = Marvin_Kine()
    kine_a.log_switch(0)
    kine_b.log_switch(0)

    ini_a = kine_a.load_config(arm_type=0, config_path=config_path)
    ini_b = kine_b.load_config(arm_type=1, config_path=config_path)
    if not ini_a or not ini_b:
        raise RuntimeError(f"failed to load config: {config_path}")

    kine_a.initial_kine(
        robot_type=ini_a['TYPE'][0],
        dh=ini_a['DH'][0],
        pnva=ini_a['PNVA'][0],
        j67=ini_a['BD'][0],
    )
    kine_b.initial_kine(
        robot_type=ini_a['TYPE'][1],
        dh=ini_a['DH'][1],
        pnva=ini_a['PNVA'][1],
        j67=ini_a['BD'][1],
    )
    return kine_a, kine_b


# ---------------------------------------------------------------------------
# IK precomputation
# ---------------------------------------------------------------------------

def _max_overshoot(joints):
    """
    Return (max_overshoot_deg, worst_joint_idx, worst_joint_value) for the
    joint that most exceeds its [JOINT_LO, JOINT_HI] limit. If all joints are
    within limits, returns (0.0, -1, 0.0).
    """
    worst = 0.0
    worst_i = -1
    worst_v = 0.0
    for i, v in enumerate(joints):
        if v < JOINT_LO[i]:
            o = JOINT_LO[i] - v
        elif v > JOINT_HI[i]:
            o = v - JOINT_HI[i]
        else:
            continue
        if o > worst:
            worst = o
            worst_i = i
            worst_v = v
    return worst, worst_i, worst_v


def _clamp_joints(joints):
    return [min(JOINT_HI[i], max(JOINT_LO[i], v)) for i, v in enumerate(joints)]


def solve_ik(kine, mat4x4, ref_joints):
    """
    Run IK and pick the IK branch (out of up to 4 returned) with the smallest
    max joint-limit overshoot. Returns (joints, overshoot) on success or
    (None, error_string) on failure.
    """
    sp = FX_InvKineSolvePara()
    sp.set_input_ik_target_tcp(kine.mat4x4_to_mat1x16(mat4x4))
    sp.set_input_ik_ref_joint(ref_joints)
    sp.set_input_ik_zsp_type(0)
    res = kine.ik(structure_data=sp)
    if not res:
        return None, "solver failed (out of range or J4 singular)"

    all_joints = res.m_OutPut_AllJoint.to_list()  # 64 floats, 8x8 row-major
    n = res.m_OutPut_Result_Num
    best_joints = None
    best_over = float('inf')
    best_idx = -1
    best_val = 0.0
    for k in range(min(n, 8)):
        j = all_joints[k * 8: k * 8 + 7]
        over, idx, val = _max_overshoot(j)
        if over < best_over:
            best_over = over
            best_joints = j
            best_idx = idx
            best_val = val

    if best_joints is None:
        return None, "no IK branches"
    if best_over > LIMIT_TOL_DEG:
        return None, (
            f"all branches violate limits (best overshoot {best_over:.2f} deg "
            f"on J{best_idx + 1}={best_val:+.2f}, limit "
            f"[{JOINT_LO[best_idx]:+.0f}, {JOINT_HI[best_idx]:+.0f}])"
        )
    return best_joints, best_over


def precompute_ik(frames, kine_a, kine_b, seed_ref):
    out = []
    ref_a = list(seed_ref)
    ref_b = list(seed_ref)
    max_over_a = 0.0
    max_over_b = 0.0
    overshoot_count_a = 0
    overshoot_count_b = 0
    first_overshoots_a = []
    first_overshoots_b = []
    for i, (t, mat_l, mat_r) in enumerate(frames):
        ja, info_a = solve_ik(kine_a, mat_l, ref_a)
        if ja is None:
            raise RuntimeError(f"left IK failed at frame {i} (t={t}): {info_a}")
        jb, info_b = solve_ik(kine_b, mat_r, ref_b)
        if jb is None:
            raise RuntimeError(f"right IK failed at frame {i} (t={t}): {info_b}")
        if info_a > 0:
            overshoot_count_a += 1
            if len(first_overshoots_a) < 3:
                idx = max(range(7), key=lambda k: max(0, JOINT_LO[k] - ja[k], ja[k] - JOINT_HI[k]))
                first_overshoots_a.append((i, t, info_a, idx + 1, ja[idx]))
        if info_b > 0:
            overshoot_count_b += 1
            if len(first_overshoots_b) < 3:
                idx = max(range(7), key=lambda k: max(0, JOINT_LO[k] - jb[k], jb[k] - JOINT_HI[k]))
                first_overshoots_b.append((i, t, info_b, idx + 1, jb[idx]))
        max_over_a = max(max_over_a, info_a)
        max_over_b = max(max_over_b, info_b)
        ref_a, ref_b = ja, jb
        out.append((t, _clamp_joints(ja), _clamp_joints(jb)))
    logger.info(f"  max joint-limit overshoot before clamping: "
                f"left={max_over_a:.2f} deg, right={max_over_b:.2f} deg "
                f"(tolerance {LIMIT_TOL_DEG} deg)")
    logger.info(f"  frames with any overshoot: left={overshoot_count_a}/{len(frames)}, "
                f"right={overshoot_count_b}/{len(frames)}")
    for label, lst in (("left", first_overshoots_a), ("right", first_overshoots_b)):
        for i, t, ov, jn, jv in lst:
            logger.info(f"    {label} first overshoot frame {i} (t={t:.4f}): "
                        f"J{jn}={jv:+.2f} (over by {ov:.2f} deg)")
    return out


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------

def connect_robot():
    dcss = DCSS()
    robot = Marvin_Robot()
    if robot.connect(ROBOT_IP) == 0:
        raise RuntimeError(f"failed to connect to {ROBOT_IP} (port occupied?)")
    time.sleep(0.5)
    frame_prev = None
    for _ in range(10):
        sub = robot.subscribe(dcss)
        fs = sub['outputs'][0]['frame_serial']
        if fs != 0 and fs != frame_prev:
            return robot, dcss
        frame_prev = fs
        time.sleep(0.1)
    robot.release_robot()
    raise RuntimeError("no data stream from robot")


def wait_for_convergence(robot, dcss, target_a, target_b, timeout=30.0, tol=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        sub = robot.subscribe(dcss)
        fb_a = sub["outputs"][0]["fb_joint_pos"]
        fb_b = sub["outputs"][1]["fb_joint_pos"]
        max_err_a = max(abs(f - t) for f, t in zip(fb_a, target_a))
        max_err_b = max(abs(f - t) for f, t in zip(fb_b, target_b))
        if max_err_a < tol and max_err_b < tol:
            return sub
        time.sleep(0.05)
    logger.warning("convergence wait timed out")
    return robot.subscribe(dcss)


def fmt(vals):
    return "[" + ", ".join(f"{v:+7.2f}" for v in vals) + "]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
speed_factor = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SPEED_FACTOR
if speed_factor <= 0:
    raise ValueError(f"speed_factor must be > 0 (got {speed_factor})")

logger.info(f"Loading trajectory: {csv_path}")
frames = load_trajectory(csv_path)
duration = frames[-1][0] - frames[0][0]
rate = (len(frames) - 1) / duration if duration > 0 else 0.0
logger.info(f"  {len(frames)} synchronized frames, "
            f"duration {duration:.2f}s, avg rate {rate:.1f} Hz")
logger.info(f"  speed factor: {speed_factor}x "
            f"(playback duration {duration / speed_factor:.2f}s)")

logger.info(f"Initializing kinematics ({CONFIG_FILE}) ...")
config_path = os.path.join(current_dir, CONFIG_FILE)
kine_a, kine_b = init_kinematics(config_path)

logger.info("Pre-computing IK for all frames ...")
joint_traj = precompute_ik(frames, kine_a, kine_b, FIRST_REF_JOINTS)
logger.info(f"  IK ok: {len(joint_traj)} frames")
logger.info(f"  first joints A: {fmt(joint_traj[0][1])}")
logger.info(f"  first joints B: {fmt(joint_traj[0][2])}")
logger.info(f"  last  joints A: {fmt(joint_traj[-1][1])}")
logger.info(f"  last  joints B: {fmt(joint_traj[-1][2])}")

logger.info(f"Connecting to robot at {ROBOT_IP} ...")
robot, dcss = connect_robot()

# Clear errors
robot.clear_set()
robot.clear_error('A')
robot.clear_error('B')
robot.send_cmd()
time.sleep(0.5)

# Enter position mode at slow speed for the initial move
robot.clear_set()
robot.set_state(arm='A', state=1)
robot.set_vel_acc(arm='A', velRatio=5, AccRatio=5)
robot.set_state(arm='B', state=1)
robot.set_vel_acc(arm='B', velRatio=5, AccRatio=5)
robot.send_cmd()
time.sleep(1.0)

first_t, first_a, first_b = joint_traj[0]

logger.info("Moving both arms to start pose (slow) ...")
robot.clear_set()
robot.set_joint_cmd_pose(arm='A', joints=first_a)
robot.set_joint_cmd_pose(arm='B', joints=first_b)
robot.send_cmd()
time.sleep(0.2)
wait_for_convergence(robot, dcss, first_a, first_b, timeout=30)
logger.info("  start pose reached")

# Bump speed for streaming so position-follow tracks the dense waypoints tightly
robot.clear_set()
robot.set_vel_acc(arm='A', velRatio=50, AccRatio=50)
robot.set_vel_acc(arm='B', velRatio=50, AccRatio=50)
robot.send_cmd()
time.sleep(0.3)

# Stream waypoints at the CSV's native cadence using a monotonic clock
logger.info(f"Streaming {len(joint_traj)} frames ...")
t_csv0 = joint_traj[0][0]
t_start = time.monotonic()
late_frames = 0
for t_csv, ja, jb in joint_traj:
    target_t = t_start + (t_csv - t_csv0) / speed_factor
    now = time.monotonic()
    sleep_for = target_t - now
    if sleep_for > 0:
        time.sleep(sleep_for)
    else:
        late_frames += 1
    robot.clear_set()
    robot.set_joint_cmd_pose(arm='A', joints=ja)
    robot.set_joint_cmd_pose(arm='B', joints=jb)
    robot.send_cmd()
elapsed = time.monotonic() - t_start
logger.info(f"  streaming done in {elapsed:.2f}s "
            f"(csv duration {joint_traj[-1][0] - t_csv0:.2f}s, "
            f"late frames: {late_frames})")

# Let the controller settle on the final commanded pose
last_t, last_a, last_b = joint_traj[-1]
wait_for_convergence(robot, dcss, last_a, last_b, timeout=10, tol=0.2)

logger.info("Disabling servos and releasing robot ...")
robot.clear_set()
robot.set_state(arm='A', state=0)
robot.set_state(arm='B', state=0)
robot.send_cmd()
time.sleep(0.5)
robot.release_robot()
logger.info("Done.")
