"""
Read-only diagnostic: verify the MuJoCo URDF and the Tianji SDK kinematics
agree on where the end-effector is for a given joint configuration.

The mink-based CSV runner computes joint targets against the URDF in
scene_tianji_wuji_full.xml. Those joints are then sent to the real robot via
set_joint_cmd_pose (rad2deg, no sign/offset conversion). If the URDF's joint
axes or zero offsets don't match the real robot's, the same joint values
produce different physical motion — which is what you'd see if the sim
moves the hands forward but the real robot moves them backward.

This script:
  1. Connects to the robot read-only, subscribes, reads current joints.
  2. Runs SDK FK (ccs_m6_40.MvKDCfg) -> SDK-local TCP pose for both arms.
  3. Converts that SDK-local TCP to world via the per-arm base transform
     derived from scene_tianji_wuji_full.xml (calvinzqiu fork).
  4. Pushes the same joints into MuJoCo sim, mj_forward, reads site_xpos.
  5. Prints both world TCPs and the delta.
  6. Also walks through each joint: temporarily perturbs it by +10 deg
     (both SDK and sim) and checks whether the resulting EE position delta
     has the same direction in both worlds. A sign mismatch on any joint
     means that joint's axis is flipped between URDF and real robot.

NO MOTION is ever commanded. The robot connection is purely for subscribe().
"""

import sys
import os
import time

import numpy as np

DEXTELE_DIR = '/home/tianji/Desktop/tianji_teleop/DexTele'
SCENE_XML = os.path.join(DEXTELE_DIR, 'robots/tianji/scene_tianji_wuji_full.xml')
sys.path.insert(0, DEXTELE_DIR)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_tianji_arm_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _tianji_arm_dir)

import mujoco  # noqa: E402
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS  # noqa: E402
from SDK_PYTHON.fx_kine import Marvin_Kine  # noqa: E402

ROBOT_IP = '192.168.1.190'
CONFIG_FILE = 'ccs_m6_40.MvKDCfg'


def say(msg=""):
    """Print to stdout (flushed) so redirects without 2>&1 still capture it."""
    print(msg, flush=True)


# calvinzqiu arm base world transforms (mirrors test_run_csv_mink.py)
def sdk_local_to_world(arm, x_l_mm, y_l_mm, z_l_mm):
    """SDK FK returns mm in arm's local DH frame. Convert to world (m)."""
    # Derived earlier in this project: the SDK's local frame ≈ the URDF's
    # arm-base frame. R_left (local->world) = [[1,0,0],[0,0,1],[0,-1,0]]
    # R_right (local->world) = [[1,0,0],[0,0,-1],[0,1,0]]
    x = x_l_mm / 1000.0
    y_l = y_l_mm / 1000.0
    z_l = z_l_mm / 1000.0
    if arm == 'left':
        xw = x + 0.0
        yw = z_l + 0.04
        zw = -y_l + 1.01
    else:
        xw = x + 0.0
        yw = -z_l - 0.04
        zw = y_l + 1.01
    return np.array([xw, yw, zw])


def fmt3(v, width=8, decimals=3):
    return "[" + ", ".join(f"{x:+{width}.{decimals}f}" for x in v) + "]"


# ---------- Init SDK kinematics for both arms ----------
say(f"Initializing SDK kinematics ({CONFIG_FILE}) ...")
config_path = os.path.join(_current_dir, CONFIG_FILE)
kine_a = Marvin_Kine()  # left = arm_type 0
kine_b = Marvin_Kine()  # right = arm_type 1
kine_a.log_switch(0)
kine_b.log_switch(0)
ini_a = kine_a.load_config(arm_type=0, config_path=config_path)
ini_b = kine_b.load_config(arm_type=1, config_path=config_path)
kine_a.initial_kine(
    robot_type=ini_a['TYPE'][0], dh=ini_a['DH'][0],
    pnva=ini_a['PNVA'][0], j67=ini_a['BD'][0])
kine_b.initial_kine(
    robot_type=ini_a['TYPE'][1], dh=ini_a['DH'][1],
    pnva=ini_a['PNVA'][1], j67=ini_a['BD'][1])

# ---------- Load MuJoCo scene ----------
say(f"Loading MuJoCo scene: {SCENE_XML}")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
left_qpos_addrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint{i}")] for i in range(1, 8)]
right_qpos_addrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_joint{i}")] for i in range(1, 8)]
left_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_ee_site")
right_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")


def mujoco_fk_world(arm, joints_deg):
    """Push the 7 joint values (deg) into MuJoCo and return the EE world pos (m)."""
    addrs = left_qpos_addrs if arm == 'left' else right_qpos_addrs
    site = left_ee_site if arm == 'left' else right_ee_site
    for i, addr in enumerate(addrs):
        data.qpos[addr] = np.deg2rad(joints_deg[i])
    mujoco.mj_forward(model, data)
    return np.array(data.site_xpos[site])


def sdk_fk_world(arm, joints_deg):
    """SDK FK -> local xyzabc -> world pos (m) using the calvinzqiu base transform."""
    kine = kine_a if arm == 'left' else kine_b
    mat = kine.fk(joints=list(joints_deg))
    xyzabc = kine.mat4x4_to_xyzabc(mat)
    return sdk_local_to_world(arm, xyzabc[0], xyzabc[1], xyzabc[2])


# ---------- Connect robot (read-only) ----------
say(f"Connecting to robot at {ROBOT_IP} (read-only, no motion will be sent) ...")
dcss = DCSS()
robot = Marvin_Robot()
if robot.connect(ROBOT_IP) == 0:
    say("Connection failed. Exiting.")
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
    say("No data stream. Exiting.")
    robot.release_robot()
    sys.exit(1)
say("Robot connected.")

sub = robot.subscribe(dcss)
cur_a = np.array(sub["outputs"][0]["fb_joint_pos"][:7], dtype=np.float64)
cur_b = np.array(sub["outputs"][1]["fb_joint_pos"][:7], dtype=np.float64)
say(f"  current joints A (left)  deg: {fmt3(cur_a, decimals=2)}")
say(f"  current joints B (right) deg: {fmt3(cur_b, decimals=2)}")

# Done with the robot — release immediately so we never even risk sending motion.
robot.release_robot()
say("  (robot released; the rest of this diagnostic is offline)")

# ---------- Compare FKs at the current pose ----------
say("\n===== Current-pose FK comparison (world frame, meters) =====")
for arm, joints in (('left', cur_a), ('right', cur_b)):
    p_sdk = sdk_fk_world(arm, joints)
    p_muj = mujoco_fk_world(arm, joints)
    d = p_muj - p_sdk
    say(f"  {arm}:")
    say(f"    SDK    world: {fmt3(p_sdk)}")
    say(f"    MuJoCo world: {fmt3(p_muj)}")
    say(f"    MuJoCo - SDK: {fmt3(d)}   |delta| = {np.linalg.norm(d)*1000:.2f} mm")

# ---------- Per-joint perturbation test ----------
DELTA_DEG = 10.0
say(f"\n===== Per-joint +{DELTA_DEG} deg perturbation "
            f"(sign mismatch = joint axis is flipped) =====")
for arm, joints in (('left', cur_a), ('right', cur_b)):
    p0_sdk = sdk_fk_world(arm, joints)
    p0_muj = mujoco_fk_world(arm, joints)
    say(f"  {arm}:")
    for j in range(7):
        q = joints.copy()
        q[j] += DELTA_DEG
        d_sdk = sdk_fk_world(arm, q) - p0_sdk
        d_muj = mujoco_fk_world(arm, q) - p0_muj
        cos = 0.0
        if np.linalg.norm(d_sdk) > 1e-9 and np.linalg.norm(d_muj) > 1e-9:
            cos = float(np.dot(d_sdk, d_muj) / (np.linalg.norm(d_sdk) * np.linalg.norm(d_muj)))
        tag = "OK    " if cos > 0.9 else "FLIP? " if cos < -0.9 else "DIFF  "
        say(f"    J{j+1}: {tag}  cos={cos:+.3f}  "
                    f"d_sdk={fmt3(d_sdk, decimals=4)}  d_muj={fmt3(d_muj, decimals=4)}")
