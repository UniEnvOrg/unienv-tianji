import sys
import os
import math
import csv
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara

logging.getLogger().setLevel(logging.CRITICAL)

DEMO2_PATH = '/home/tianji/Downloads/demo2.csv'
EE_TEST_PATH = '/home/tianji/Desktop/tianji_teleop/ee_test_poses_tianji_sdk.csv'
CONFIG_FILE = 'ccs_m6_40.MvKDCfg'

# From calvinzqiu/DexTele robots/tianji/scene_tianji_wuji_full.xml (the fork
# that's actually being run):
#   <body name="left_base_link"  pos="0.0  0.04 1.01" quat="0.707105 -0.707108 0 0">
#   <body name="right_base_link" pos="0.0 -0.04 1.01" quat="0.707105  0.707108 0 0">
# The two arms are MIRRORED about the X-Z plane (different quaternions).
#
# Left  R = [[1,0,0],[0,0,1],[0,-1,0]]   -> local +X=world +X, +Y=world -Z, +Z=world +Y
# Right R = [[1,0,0],[0,0,-1],[0,1,0]]   -> local +X=world +X, +Y=world +Z, +Z=world -Y
# T_world_armbase^-1 = [R^T, -R^T @ t; 0 1]
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

SEEDS = {
    'avp_start':    [  0.0, -60.0,   0.0, -90.0,    0.0,   0.0,    0.0],
    'mid_bend':     [  0.0, -45.0,   0.0, -90.0,    0.0,   0.0,    0.0],
    'right_lean':   [-30.0, -60.0,   0.0, -90.0,    0.0,   0.0,    0.0],
    'left_lean':    [ 30.0, -60.0,   0.0, -90.0,    0.0,   0.0,    0.0],
    'left_yaw_far': [120.0, -70.0, -25.0, -80.0,  100.0,  20.0,   40.0],
    'right_yaw_far':[-120.0,-70.0,  25.0, -80.0, -100.0, -20.0,  -40.0],
    'reach_fwd':    [  0.0, -90.0,   0.0, -45.0,    0.0,   0.0,    0.0],
}

# CCS joint limits (deg) — see test_all_joints_position.py
JOINT_LO = [-170, -120, -170, -145, -170, -60, -90]
JOINT_HI = [ 170,  120,  170,   60,  170,  60,  90]


# ----- Pure-Python 4x4 helpers -----

def mat_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def quat_pos_to_mat4x4(x_mm, y_mm, z_mm, qw, qx, qy, qz):
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return [
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),     x_mm],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),     y_mm],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy), z_mm],
        [0.0,             0.0,             0.0,             1.0],
    ]


def world_to_arm_local(arm_label, x_m, y_m, z_m, qw, qx, qy, qz):
    """Build T_world_tcp from CSV row, then return T_armlocal_tcp."""
    T_world_tcp = quat_pos_to_mat4x4(
        x_m * 1000.0, y_m * 1000.0, z_m * 1000.0, qw, qx, qy, qz
    )
    return mat_mul(ARM_BASE_T_INV[arm_label], T_world_tcp)


# ----- SDK helpers -----

def init_kine(arm_type):
    k = Marvin_Kine()
    k.log_switch(0)
    cfg = os.path.join(current_dir, CONFIG_FILE)
    ini = k.load_config(arm_type=arm_type, config_path=cfg)
    k.initial_kine(
        robot_type=ini['TYPE'][0],
        dh=ini['DH'][0],
        pnva=ini['PNVA'][0],
        j67=ini['BD'][0],
    )
    return k


def violates(joints):
    """Return list of (joint_index, value, lo, hi) for joints outside hard limits."""
    bad = []
    for i, v in enumerate(joints):
        if v < JOINT_LO[i] or v > JOINT_HI[i]:
            bad.append((i, v, JOINT_LO[i], JOINT_HI[i]))
    return bad


def try_ik(kine, mat4x4, ref):
    """
    Returns:
        (joints, status_str, max_overshoot)
    where joints is the best-among-4-branches solution (least max overshoot),
    status_str is "ok" / "limit overshoot Xdeg" / "out of range" / "solver failed".
    """
    sp = FX_InvKineSolvePara()
    sp.set_input_ik_target_tcp(kine.mat4x4_to_mat1x16(mat4x4))
    sp.set_input_ik_ref_joint(ref)
    sp.set_input_ik_zsp_type(0)
    res = kine.ik(structure_data=sp)
    if not res:
        return None, "solver failed (or out of range / J4 singular)", None

    # Iterate all branches; pick the one with least overshoot
    all_joints = res.m_OutPut_AllJoint.to_list()  # 64-flat
    n = res.m_OutPut_Result_Num
    best = None
    best_over = float('inf')
    for k in range(min(n, 8)):
        j = all_joints[k * 8: k * 8 + 7]
        bad = violates(j)
        over = 0.0 if not bad else max(
            max(0.0, lo - v, v - hi) for _, v, lo, hi in bad
        )
        if over < best_over:
            best_over = over
            best = j

    if best is None:
        return None, "no branches", None
    if best_over == 0.0:
        return best, "ok", 0.0
    return best, f"limit overshoot {best_over:.1f}deg", best_over


def first_row(csv_path, arm):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['arm'].strip().lower() == arm:
                return row
    return None


def fmt_joints(j):
    return "[" + ", ".join(f"{v:+6.1f}" for v in j) + "]"


def fmt_xyz(x, y, z):
    return f"X={x:+8.1f} Y={y:+8.1f} Z={z:+8.1f}"


# ----- Tests -----

def show_workspace(kine_a, kine_b):
    print("\nFK at each seed (arm-local mm):")
    for name, j in SEEDS.items():
        fa = kine_a.mat4x4_to_xyzabc(kine_a.fk(joints=j))
        fb = kine_b.mat4x4_to_xyzabc(kine_b.fk(joints=j))
        print(f"  {name:11s} A: {fmt_xyz(fa[0], fa[1], fa[2])}   "
              f"B: {fmt_xyz(fb[0], fb[1], fb[2])}")


def report_ik(label, kine, mat4x4):
    print(f"    {label}")
    best_overall = None
    best_over = float('inf')
    for sname, sref in SEEDS.items():
        ja, status, over = try_ik(kine, mat4x4, sref)
        if ja is None:
            print(f"      seed={sname:14s} -> {status}")
        else:
            print(f"      seed={sname:14s} -> {status:25s} {fmt_joints(ja)}")
            if over is not None and over < best_over:
                best_over = over
                best_overall = ja
    return best_overall, best_over


def test_demo2_with_transform(kine_a, kine_b):
    print("\n" + "=" * 78)
    print(f"TARGET: {DEMO2_PATH} (world->arm-local transform applied)")

    for arm_label, kine in (('left', kine_a), ('right', kine_b)):
        row = first_row(DEMO2_PATH, arm_label)
        if not row:
            continue
        x_m = float(row['x_m']); y_m = float(row['y_m']); z_m = float(row['z_m'])
        qw = float(row['qw']); qx = float(row['qx']); qy = float(row['qy']); qz = float(row['qz'])

        print(f"\n  {arm_label}: world pos (mm) = ({x_m*1000:+.1f},{y_m*1000:+.1f},{z_m*1000:+.1f})")

        T_local = world_to_arm_local(arm_label, x_m, y_m, z_m, qw, qx, qy, qz)
        local_xyzabc = kine.mat4x4_to_xyzabc(T_local)
        print(f"    arm-local pos (mm) = ({local_xyzabc[0]:+.1f},{local_xyzabc[1]:+.1f},{local_xyzabc[2]:+.1f})  "
              f"abc (deg) = ({local_xyzabc[3]:+.1f},{local_xyzabc[4]:+.1f},{local_xyzabc[5]:+.1f})")
        dist = math.sqrt(local_xyzabc[0]**2 + local_xyzabc[1]**2 + local_xyzabc[2]**2)
        print(f"    arm-local distance from shoulder = {dist:.1f} mm")
        report_ik(f"(real orientation)", kine, T_local)

        # Position only, orientation held
        ref_xyzabc = kine.mat4x4_to_xyzabc(kine.fk(joints=SEEDS['avp_start']))
        held = [local_xyzabc[0], local_xyzabc[1], local_xyzabc[2],
                ref_xyzabc[3], ref_xyzabc[4], ref_xyzabc[5]]
        T_held = kine.xyzabc_to_mat4x4(held)
        report_ik(f"(held orientation)", kine, T_held)


def test_cross_arm(kine_a, kine_b):
    """Both arms have identical local kinematics. Try the right-arm position
    on the LEFT arm to confirm the position itself is reachable."""
    print("\n" + "=" * 78)
    print("CROSS-ARM SANITY: try right's local target on the LEFT arm "
          "(both arms have identical kine in local frame)")

    row = first_row(DEMO2_PATH, 'right')
    if not row:
        print("  no right row")
        return
    x_m = float(row['x_m']); y_m = float(row['y_m']); z_m = float(row['z_m'])
    qw = float(row['qw']); qx = float(row['qx']); qy = float(row['qy']); qz = float(row['qz'])
    T_local = world_to_arm_local('right', x_m, y_m, z_m, qw, qx, qy, qz)
    local_xyzabc = kine_a.mat4x4_to_xyzabc(T_local)
    print(f"  right local pos (mm) = ({local_xyzabc[0]:+.1f},{local_xyzabc[1]:+.1f},{local_xyzabc[2]:+.1f})")
    print(f"  feeding to LEFT arm IK (real orientation):")
    report_ik("on-A:", kine_a, T_local)
    print(f"  feeding to LEFT arm IK (held orientation):")
    ref_xyzabc = kine_a.mat4x4_to_xyzabc(kine_a.fk(joints=SEEDS['avp_start']))
    held = [local_xyzabc[0], local_xyzabc[1], local_xyzabc[2],
            ref_xyzabc[3], ref_xyzabc[4], ref_xyzabc[5]]
    report_ik("on-A:", kine_a, kine_a.xyzabc_to_mat4x4(held))


print("=" * 78)
print(f"Loading kinematics ({CONFIG_FILE}) ...")
kine_a = init_kine(0)
kine_b = init_kine(1)
show_workspace(kine_a, kine_b)
test_demo2_with_transform(kine_a, kine_b)
test_cross_arm(kine_a, kine_b)
