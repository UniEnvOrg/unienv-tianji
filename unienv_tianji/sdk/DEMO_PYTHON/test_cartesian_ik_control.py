import sys
import os
import time
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
current_path = current_dir

from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS
from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara

"""
Cartesian IK control demo for left arm (Arm A) at 50Hz control frequency.

Uses movLA to plan a straight-line Cartesian trajectory, then streams
the planned waypoints to the robot at 50Hz (20ms intervals) under
Cartesian impedance control.

Steps:
    1. Connect robot, clear errors
    2. Initialize kinematics for left arm
    3. Move left arm to starting pose [0, -60, 0, -90, 0, 0, 0] (position mode)
    4. Switch to torque mode with Cartesian impedance
    5. FK to compute end-effector XYZABC
    6. movLA to plan +100mm X trajectory at 50Hz
    7. Stream waypoints at 50Hz (20ms per point)
    8. Verify end-effector displacement
    9. Return to start, cleanup

Config note: uses ccs_m6_31.MvKDCfg — change if your robot is a different model.
"""

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('ik_test')
logger.setLevel(logging.DEBUG)

CTRL_FREQ_HZ = 1000
CTRL_PERIOD_S = 1.0 / CTRL_FREQ_HZ  # 20ms


def wait_for_motion(robot, dcss, timeout=20.0, tol=0.5):
    """Wait until arm A joints stop moving."""
    t0 = time.time()
    prev = None
    while time.time() - t0 < timeout:
        sub = robot.subscribe(dcss)
        cur = sub["outputs"][0]["fb_joint_pos"]
        if prev is not None:
            if max(abs(c - p) for c, p in zip(cur, prev)) < tol:
                return sub
        prev = cur
        time.sleep(0.2)
    logger.warning("Motion wait timed out")
    return robot.subscribe(dcss)


def fmt(vals, decimals=3):
    return "[" + ", ".join(f"{v:+.{decimals}f}" for v in vals) + "]"


# ==================== Initialize Kinematics ====================
logger.info("Initializing kinematics for left arm (CCS M6 3.1) ...")
kine = Marvin_Kine()
kine.log_switch(0)

config_path = os.path.join(current_path, 'ccs_m6_31.MvKDCfg')
ini = kine.load_config(arm_type=0, config_path=config_path)
kine.initial_kine(
    robot_type=ini['TYPE'][0],
    dh=ini['DH'][0],
    pnva=ini['PNVA'][0],
    j67=ini['BD'][0]
)
logger.info("Kinematics initialized.\n")

# ==================== Connect Robot ====================
dcss = DCSS()
robot = Marvin_Robot()

logger.info("Connecting to robot at 192.168.1.190 ...")
if robot.connect('192.168.1.190') == 0:
    logger.error("Connection failed. Exiting.")
    sys.exit(1)
time.sleep(0.5)

# Verify data stream
frame_prev = None
connected = False
for _ in range(10):
    sub = robot.subscribe(dcss)
    fs = sub['outputs'][0]['frame_serial']
    if fs != 0 and fs != frame_prev:
        connected = True
        frame_prev = fs
    time.sleep(0.1)
if not connected:
    logger.error("No data from robot. Exiting.")
    robot.release_robot()
    sys.exit(1)
logger.info("Robot connected.\n")

# Clear errors
robot.clear_set()
robot.clear_error('A')
robot.clear_error('B')
robot.send_cmd()
time.sleep(0.5)

# ==================== Step 1: Move to Starting Pose (position mode) ====================
start_joints = [0.0, -60.0, 0.0, -90.0, 0.0, 0.0, 0.0]

robot.clear_set()
robot.set_state(arm='A', state=1)  # position mode
robot.set_vel_acc(arm='A', velRatio=5, AccRatio=5)
robot.set_state(arm='B', state=0)  # keep B arm disabled
robot.send_cmd()
time.sleep(1)

logger.info("Step 1: Moving left arm to starting pose (position mode)")
logger.info(f"  Target joints: {fmt(start_joints)}")

robot.clear_set()
robot.set_joint_cmd_pose(arm='A', joints=start_joints)
robot.send_cmd()
sub = wait_for_motion(robot, dcss)

actual_joints = list(sub["outputs"][0]["fb_joint_pos"])
logger.info(f"  Reached joints: {fmt(actual_joints)}")

# ==================== Step 2: FK — Compute End-Effector Pose ====================
fk_mat = kine.fk(joints=actual_joints)
start_xyzabc = kine.mat4x4_to_xyzabc(fk_mat)

logger.info(f"\nStep 2: FK at starting pose")
logger.info(f"  End-effector: X={start_xyzabc[0]:.2f}, Y={start_xyzabc[1]:.2f}, Z={start_xyzabc[2]:.2f} mm")
logger.info(f"  Orientation:  A={start_xyzabc[3]:.2f}, B={start_xyzabc[4]:.2f}, C={start_xyzabc[5]:.2f} deg")

# ==================== Step 3: Switch to Cartesian Impedance Mode ====================
logger.info(f"\nStep 3: Switching to Cartesian impedance mode")

# Set impedance parameters (stiffness K, damping D)
robot.clear_set()
robot.set_cart_kd_params(
    arm='A',
    K=[8000, 8000, 8000, 600, 600, 600, 20],
    D=[0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1],
    type=2
)
robot.send_cmd()
time.sleep(0.5)

# Switch to torque mode with Cartesian impedance
robot.clear_set()
robot.set_state(arm='A', state=3)       # torque mode
robot.set_impedance_type(arm='A', type=2)  # Cartesian impedance
robot.set_vel_acc(arm='A', velRatio=50, AccRatio=50)
robot.send_cmd()
time.sleep(0.5)

# Verify mode switch
sub = robot.subscribe(dcss)
cur_state = sub['states'][0]['cur_state']
imp_type = sub['inputs'][0]['imp_type']
logger.info(f"  State={cur_state} (3=torque), Impedance type={imp_type} (2=Cartesian)")

# ==================== Step 4: Plan +100mm X trajectory at 50Hz ====================
target_xyzabc = start_xyzabc.copy()
target_xyzabc[0] += 100.0  # +100mm along X axis

logger.info(f"\nStep 4: Planning trajectory with movLA at {CTRL_FREQ_HZ}Hz")
logger.info(f"  Start X={start_xyzabc[0]:.2f} mm -> Target X={target_xyzabc[0]:.2f} mm")

points, pset = kine.movLA(
    start_xyzabc=start_xyzabc,
    end_xyzabc=target_xyzabc,
    ref_joints=actual_joints,
    vel=20,     # mm/s (slow)
    acc=20,     # mm/s^2
    freq_hz=CTRL_FREQ_HZ
)
logger.info(f"  Planned {len(points)} waypoints ({len(points) * CTRL_PERIOD_S:.2f}s trajectory)")
if points:
    logger.info(f"  First waypoint: {fmt(points[0])}")
    logger.info(f"  Last  waypoint: {fmt(points[-1])}")

# ==================== Step 5: Stream waypoints at 50Hz ====================
logger.info(f"\nStep 5: Streaming {len(points)} waypoints at {CTRL_FREQ_HZ}Hz ...")

t_start = time.time()
for i, wp in enumerate(points):
    t_loop = time.time()
    robot.clear_set()
    robot.set_joint_cmd_pose(arm='A', joints=wp)
    robot.send_cmd()
    # Maintain 50Hz timing
    elapsed = time.time() - t_loop
    sleep_time = CTRL_PERIOD_S - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
t_elapsed = time.time() - t_start
logger.info(f"  Streaming done. Elapsed: {t_elapsed:.2f}s")

# Wait for settling
time.sleep(0.5)

# ==================== Step 6: Verify end-effector position ====================
sub = robot.subscribe(dcss)
reached_joints = list(sub["outputs"][0]["fb_joint_pos"])
reached_mat = kine.fk(joints=reached_joints)
reached_xyzabc = kine.mat4x4_to_xyzabc(reached_mat)
x_delta = reached_xyzabc[0] - start_xyzabc[0]

logger.info(f"\nStep 6: Verification")
logger.info(f"  Reached joints: {fmt(reached_joints)}")
logger.info(f"  Reached: X={reached_xyzabc[0]:.2f}, Y={reached_xyzabc[1]:.2f}, Z={reached_xyzabc[2]:.2f} mm")
logger.info(f"  X displacement: {x_delta:+.2f} mm (target: +100.00 mm)")

# ==================== Step 7: Move to joint-control-script end pose via IK ====================
# Target: the [18,18,18,18,18,18,18] deg pose from test_all_joints_position.py
joint_target_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# FK to get the target Cartesian pose
target2_mat = kine.fk(joints=joint_target_ref)
target2_xyzabc = kine.mat4x4_to_xyzabc(target2_mat)

# IK to verify reachability from current config and solve target joints
sp2 = FX_InvKineSolvePara()
mat16_2 = kine.mat4x4_to_mat1x16(target2_mat)
sp2.set_input_ik_target_tcp(mat16_2)
sp2.set_input_ik_ref_joint(joint_target_ref)
sp2.set_input_ik_zsp_type(0)
ik2 = kine.ik(structure_data=sp2)

ik_target_joints = ik2.m_Output_RetJoint.to_list() if ik2 else joint_target_ref

logger.info(f"\nStep 7: Move to joint-script end pose [18]*7 deg (joint-space interpolation)")
logger.info(f"  Current EE: X={reached_xyzabc[0]:.2f}, Y={reached_xyzabc[1]:.2f}, Z={reached_xyzabc[2]:.2f} mm")
logger.info(f"  Target  EE: X={target2_xyzabc[0]:.2f}, Y={target2_xyzabc[1]:.2f}, Z={target2_xyzabc[2]:.2f} mm")
logger.info(f"  IK target joints: {fmt(ik_target_joints)}")
if ik2:
    logger.info(f"  IK out of range: {ik2.m_Output_IsOutRange}, joint exceeded: {ik2.m_Output_IsJntExd}")

# Linear interpolation in joint space at CTRL_FREQ_HZ
# Duration: based on max joint displacement at a safe speed (~30 deg/s)
max_delta = max(abs(t - c) for t, c in zip(ik_target_joints, reached_joints))
duration = max(max_delta / 10.0, 1.0)  # ~10 deg/s, slow
n_points = max(int(duration * CTRL_FREQ_HZ), 2)

logger.info(f"  Max joint delta: {max_delta:.1f} deg, duration: {duration:.2f}s, {n_points} waypoints")

# Generate joint-space interpolation waypoints
waypoints = []
for i in range(n_points):
    alpha = (i + 1) / n_points  # 0 -> 1
    wp = [c + alpha * (t - c) for c, t in zip(reached_joints, ik_target_joints)]
    waypoints.append(wp)

# Stream at CTRL_FREQ_HZ
logger.info(f"  Streaming {n_points} waypoints at {CTRL_FREQ_HZ}Hz ...")
t_start = time.time()
for wp in waypoints:
    t_loop = time.time()
    robot.clear_set()
    robot.set_joint_cmd_pose(arm='A', joints=wp)
    robot.send_cmd()
    elapsed = time.time() - t_loop
    sleep_time = CTRL_PERIOD_S - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
t_elapsed = time.time() - t_start
logger.info(f"  Streaming done. Elapsed: {t_elapsed:.2f}s")

time.sleep(0.5)

# Verify
sub = robot.subscribe(dcss)
final_joints = list(sub["outputs"][0]["fb_joint_pos"])
final_mat = kine.fk(joints=final_joints)
final_xyzabc = kine.mat4x4_to_xyzabc(final_mat)
logger.info(f"\nStep 8: Verification")
logger.info(f"  Target  joints: {fmt(joint_target_ref)}")
logger.info(f"  Reached joints: {fmt(final_joints)}")
logger.info(f"  Reached EE: X={final_xyzabc[0]:.2f}, Y={final_xyzabc[1]:.2f}, Z={final_xyzabc[2]:.2f} mm")
logger.info(f"  Target  EE: X={target2_xyzabc[0]:.2f}, Y={target2_xyzabc[1]:.2f}, Z={target2_xyzabc[2]:.2f} mm")

# ==================== Cleanup ====================
logger.info(f"\nCleanup: disabling servos and releasing robot ...")
robot.clear_set()
robot.set_state(arm='A', state=0)
robot.send_cmd()
time.sleep(0.5)
robot.release_robot()
logger.info("Done. Robot released.")
