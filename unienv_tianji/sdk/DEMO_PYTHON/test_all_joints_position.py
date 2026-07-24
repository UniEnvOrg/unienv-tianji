import sys
import os
import time
import math
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS

"""
All-joint position control test for bi-arm CCS Marvin robot.

Steps:
    1. Connect to robot, clear errors
    2. Set position mode at low speed (5% vel, 5% acc)
    3. Reset both arms to zero posture
    4. Wait for motion to complete
    5. Move all 7 joints by pi/10 (18 deg), clamped to joint limits
    6. Wait for motion to complete
    7. Reset back to zero posture
    8. Disable servos and release

Joint limits (CCS):
    J1: [-170, 170], J2: [-120, 120], J3: [-170, 170],
    J4: [-145,  60], J5: [-170, 170], J6: [ -60,  60], J7: [-90, 90]
"""

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)

# CCS joint limits in degrees
JOINT_MIN = [-170, -120, -170, -145, -170, -60, -90]
JOINT_MAX = [ 170,  120,  170,   60,  170,  60,  90]
DELTA_DEG = math.degrees(math.pi / 10)  # 18 degrees


def clamp_joints(joints):
    """Clamp each joint to its [min, max] range."""
    return [max(lo, min(hi, j)) for j, lo, hi in zip(joints, JOINT_MIN, JOINT_MAX)]


def wait_for_convergence(robot, dcss, target_a, target_b, timeout=30.0, tol=0.1):
    """
    Wait until both arms' feedback joint positions reach the given targets
    within tol (degrees), or until timeout.
    """
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
    logger.warning("Convergence wait timed out")
    return robot.subscribe(dcss)


def print_joint_status(sub, label=""):
    """Print current joint positions for both arms."""
    if label:
        logger.info(f"\n===== {label} =====")
    for idx, name in enumerate(["A (left)", "B (right)"]):
        fb = sub["outputs"][idx]["fb_joint_pos"]
        cmd = sub["inputs"][idx]["joint_cmd_pos"]
        state = sub["states"][idx]["cur_state"]
        err = sub["states"][idx]["err_code"]
        pos_str = ", ".join(f"{v:+8.3f}" for v in fb)
        logger.info(f"  Arm {name}: state={state} err={err}")
        logger.info(f"    cmd  = [{', '.join(f'{v:+8.3f}' for v in cmd)}]")
        logger.info(f"    fb   = [{pos_str}]")


# ---------- Main ----------
dcss = DCSS()
robot = Marvin_Robot()

# 1. Connect
logger.info("Connecting to robot at 192.168.1.190 ...")
init = robot.connect('192.168.1.190')
if init == 0:
    logger.error("Connection failed (port occupied). Exiting.")
    sys.exit(1)
time.sleep(0.5)

# Verify data stream
motion_tag = 0
frame_prev = None
for _ in range(10):
    sub = robot.subscribe(dcss)
    fs = sub['outputs'][0]['frame_serial']
    if fs != 0 and fs != frame_prev:
        motion_tag += 1
        frame_prev = fs
    time.sleep(0.1)
if motion_tag == 0:
    logger.error("No data from robot. Exiting.")
    robot.release_robot()
    sys.exit(1)
logger.info("Robot connected and streaming data.")

# 2. Clear errors
robot.clear_set()
robot.clear_error('A')
robot.clear_error('B')
robot.send_cmd()
time.sleep(0.5)

# 3. Read current joint positions
sub = robot.subscribe(dcss)
cur_a = sub["outputs"][0]["fb_joint_pos"]
cur_b = sub["outputs"][1]["fb_joint_pos"]
print_joint_status(sub, "INITIAL POSTURE")

# 4. Set position mode, low speed (5% vel, 5% acc) for safety
robot.clear_set()
robot.set_state(arm='A', state=1)
robot.set_vel_acc(arm='A', velRatio=5, AccRatio=5)
robot.set_state(arm='B', state=1)
robot.set_vel_acc(arm='B', velRatio=5, AccRatio=5)
robot.send_cmd()
time.sleep(1)

# 5. Reset to zero posture
logger.info("\n>>> Step 1: Resetting both arms to zero posture ...")
zero_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
robot.clear_set()
robot.set_joint_cmd_pose(arm='A', joints=zero_joints)
robot.set_joint_cmd_pose(arm='B', joints=zero_joints)
robot.send_cmd()
time.sleep(0.2)
sub = wait_for_convergence(robot, dcss, zero_joints, zero_joints, timeout=30)
print_joint_status(sub, "AFTER RESET TO ZERO")

# 6. Move all joints by +pi/10 (18 deg), clamped to limits
target = clamp_joints([DELTA_DEG] * 7)
target_str = ", ".join(f"{v:.1f}" for v in target)
logger.info(f"\n>>> Step 2: Moving all joints by +pi/10 = +{DELTA_DEG:.1f} deg")
logger.info(f"    Target (clamped): [{target_str}]")

robot.clear_set()
robot.set_joint_cmd_pose(arm='A', joints=target)
robot.set_joint_cmd_pose(arm='B', joints=target)
robot.send_cmd()
time.sleep(0.2)
sub = wait_for_convergence(robot, dcss, target, target, timeout=30)
print_joint_status(sub, "AFTER MOVING +pi/10")

# 7. Reset back to zero
logger.info("\n>>> Step 3: Returning both arms to zero posture ...")
robot.clear_set()
robot.set_joint_cmd_pose(arm='A', joints=zero_joints)
robot.set_joint_cmd_pose(arm='B', joints=zero_joints)
robot.send_cmd()
time.sleep(0.2)
sub = wait_for_convergence(robot, dcss, zero_joints, zero_joints, timeout=30)
print_joint_status(sub, "AFTER RETURN TO ZERO")

# 8. Disable servos and release
logger.info("\n>>> Cleanup: disabling servos and releasing robot ...")
robot.clear_set()
robot.set_state(arm='A', state=0)
robot.set_state(arm='B', state=0)
robot.send_cmd()
time.sleep(0.5)
robot.release_robot()
logger.info("Done. Robot released.")
