"""
Run /home/tianji/Downloads/demo2_shoulder.csv on the real Marvin robot using
DexTele's mink IK (TianjiDualArmIKMink from teleop/tianji_arm_ik_mink.py).

CSV is always assumed to be **shoulder-local** (one entry per arm in its own
arm-base frame). It is converted to world frame using the arm base transforms
from scene_tianji_wuji_full.xml, then fed to mink one frame at a time.

Offline precompute pattern:
    1. Seed sim qpos to ARM_DEFAULT_Q (matches DexTele's init pose).
    2. For each frame: sync_configuration(data), solve to target, copy the
       IK result back into data.qpos, mj_forward to refresh derived state.
       Next frame's sync_configuration reads the updated data.qpos.
    3. A warmup loop against frame 0 runs before playback so the sim settles
       near the start pose.

We push IK results directly into data.qpos rather than using physics stepping
because we're precomputing a command trajectory — the real robot's own servos
will apply the physics dynamics when we stream the joints later.
"""

import sys
import os
import time
import math
import csv
import logging

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEXTELE_DIR = '/home/tianji/Desktop/tianji_teleop/DexTele'
SCENE_XML = os.path.join(DEXTELE_DIR, 'robots/tianji/scene_tianji_arms_with_hands_fixed.xml')

sys.path.insert(0, DEXTELE_DIR)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_tianji_arm_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _tianji_arm_dir)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
from teleop.tianji_arm_ik_mink import TianjiDualArmIKMink  # noqa: E402
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH = '/home/tianji/Downloads/demo2_shoulder.csv'
DEFAULT_SPEED_FACTOR = 1.0
ROBOT_IP = '192.168.1.190'

# Match record_bimanual_ee_demo.py solve() arguments
IK_DT = 1.0 / 60.0
IK_MAX_ITERS = 5

# Streaming: linearly interpolate the 60 Hz IK results up to this rate before
# sending to the robot. Position-mode tracking looks jittery at 60 Hz because
# each per-frame joint delta (~1-2 deg) is small enough that the firmware
# ramps then sits idle until the next command. Upsampling to 1 kHz makes
# the per-step delta tiny and the motion smooth — same trick as
# test_cartesian_ik_control.py (CTRL_FREQ_HZ = 1000).
STREAM_FREQ_HZ = 1000
STREAM_PERIOD_S = 1.0 / STREAM_FREQ_HZ

# Mirror commanded joints in a MuJoCo passive viewer alongside the real-robot
# streaming. Redraw every VIEWER_DIVISOR streaming frames so the viewer runs
# at ~60 Hz while the robot still streams at 1 kHz. Closing the viewer window
# during playback aborts the stream cleanly and drops through to the cleanup
# path (disable servos + release).
VIEWER_DIVISOR = 16

# Collision safety check: after precompute, walk every frame and check for
# arm/hand <-> prism, left-arm/hand <-> right-arm/hand, and same-side upper
# arm (link1..link4) <-> own hand. If any hit is found, abort before sending
# any motion to the robot. mujoco.mj_geomDistance ignores contype/conaffinity,
# so the visual-only arm meshes (contype=0) still participate.
TOLERANCE_PRISM_M = 0.01         # 1 cm safety margin around the rectangular prism
TOLERANCE_OTHER_M = 0.005        # 5 mm for arm<->arm / arm<->hand pairs
COLLISION_CHECK = True           # set False to skip the check

# Posture bias (same as record_bimanual_ee_demo.py ARM_DEFAULT_Q).
# Elbows bent forward, J1 pushed outward to keep arms from crossing midplane.
ARM_DEFAULT_Q = {
    'left':  np.array([-2.8, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0]),
    'right': np.array([ 2.8, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0]),
}

# Warmup: iterate mink toward the first CSV frame so the sim configuration
# settles before playback starts. We run at most WARMUP_MAX_ITERS but stop
# early as soon as the EE residual stops improving by WARMUP_TOL_MM for
# WARMUP_STALL_ITERS consecutive iterations — the residual plateaus very
# quickly against the IK's hard constraints, and extra iterations just
# expose us to rare numerical blowups in the QP solver.
WARMUP_MAX_ITERS = 150
WARMUP_STALL_ITERS = 20
WARMUP_TOL_MM = 0.5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('mink_runner')
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# CSV loader (shoulder-local -> world)
# ---------------------------------------------------------------------------

def _shoulder_to_world(arm, x_l, y_l, z_l, qw, qx, qy, qz):
    """
    Inverse of the calvinzqiu arm-base transform (shoulder_xyz=+-40mm, z=1.01m,
    quat=(0.707, -+0.707, 0, 0)):
      Left  R = [[1,0,0],[0,0,1],[0,-1,0]]   (local +Y = world -Z, local +Z = world +Y)
      Right R = [[1,0,0],[0,0,-1],[0,1,0]]   (local +Y = world +Z, local +Z = world -Y)
    world = R @ local + pos_arm_base
    world_quat = R_arm_base_quat * local_quat
    """
    if arm == 'left':
        xw = x_l
        yw = z_l + 0.04
        zw = -y_l + 1.01
        # R_arm_base = rotX(-90deg). Hamilton quat for that = (cos(-45), sin(-45), 0, 0) = (0.7071, -0.7071, 0, 0)
        # Multiply (w1+x1i+y1j+z1k) * (w2+x2i+y2j+z2k):
        w1, x1, y1, z1 = 0.7071067811865476, -0.7071067811865476, 0.0, 0.0
    else:
        xw = x_l
        yw = -z_l - 0.04
        zw = y_l + 1.01
        w1, x1, y1, z1 = 0.7071067811865476, 0.7071067811865476, 0.0, 0.0
    w2, x2, y2, z2 = qw, qx, qy, qz
    qw_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    qx_w = w1*x2 + x1*w2 + y1*z2 - z1*y2
    qy_w = w1*y2 - x1*z2 + y1*w2 + z1*x2
    qz_w = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return (xw, yw, zw, qw_w, qx_w, qy_w, qz_w)


def load_trajectory(csv_path):
    """
    Load demo2_shoulder.csv (shoulder-local per-arm poses) and return a list
    of (t, left_pos_world, left_quat_world_wxyz, right_pos_world, right_quat_world_wxyz).
    """
    by_time = {}
    with open(csv_path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            t = round(float(row['time_s']), 6)
            arm = row['arm'].strip().lower()
            if arm not in ('left', 'right'):
                raise ValueError(f"unknown arm '{arm}' in row: {row}")
            x = float(row['x_m']); y = float(row['y_m']); z = float(row['z_m'])
            qw = float(row['qw']); qx = float(row['qx']); qy = float(row['qy']); qz = float(row['qz'])
            n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if n == 0:
                raise ValueError(f"zero quat at t={t}")
            qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
            by_time.setdefault(t, {})[arm] = (x, y, z, qw, qx, qy, qz)

    frames = []
    for t in sorted(by_time):
        e = by_time[t]
        if 'left' not in e or 'right' not in e:
            raise ValueError(f"frame at t={t} missing left or right")
        lx, ly, lz, lqw, lqx, lqy, lqz = _shoulder_to_world('left',  *e['left'])
        rx, ry, rz, rqw, rqx, rqy, rqz = _shoulder_to_world('right', *e['right'])
        frames.append((
            t,
            np.array([lx, ly, lz]),
            np.array([lqw, lqx, lqy, lqz]),
            np.array([rx, ry, rz]),
            np.array([rqw, rqx, rqy, rqz]),
        ))
    return frames


# ---------------------------------------------------------------------------
# Sim setup + IK loop (matches record_bimanual_ee_demo.py)
# ---------------------------------------------------------------------------

def build_sim_and_ik():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    ik = TianjiDualArmIKMink(model, data)
    if ik.left is None or ik.right is None:
        raise RuntimeError("scene is missing left or right arm bodies")
    # Seed qpos to ARM_DEFAULT_Q and register it as the posture-task target
    for side, q in ARM_DEFAULT_Q.items():
        solver = ik.left if side == 'left' else ik.right
        for i, addr in enumerate(solver.qpos_addrs):
            data.qpos[addr] = q[i]
        solver.set_posture_target(q)
    mujoco.mj_forward(model, data)
    return model, data, ik


class IKSolveError(Exception):
    """Raised when mink's solver fails on a given target (e.g. numpy/daqp blowup)."""


def step_ik_once(model, data, ik, left_target, right_target):
    """
    One mink IK iteration on both arms. Pattern:
        for each arm:
            sync_configuration(data)  -> copy data.qpos into mink.Configuration
            solve(target)             -> advance config internally, return q_ik
            write q_ik back into data.qpos
        mj_forward(model, data)       -> refresh site_xpos etc. from new qpos

    Raises IKSolveError on a failed solve OR a non-finite result. Callers can
    catch this to hold the previous joint pose rather than abort the whole
    trajectory — mink's QP is occasionally tripped up by what appears to be a
    numpy 2.x / daqp heisenbug inside _arrays_for_stack_dispatcher.
    """
    for side, (pos, quat) in (('left', left_target), ('right', right_target)):
        solver = ik.left if side == 'left' else ik.right
        solver.sync_configuration(data)
        try:
            q_ik = solver.solve(pos, target_quat=quat, dt=IK_DT, max_iters=IK_MAX_ITERS)
        except Exception as e:
            raise IKSolveError(
                f"{side} solver raised {type(e).__name__}: {e}"
            ) from e
        if not np.all(np.isfinite(q_ik)):
            raise IKSolveError(
                f"{side} IK returned non-finite joints: {q_ik}"
            )
        for i, addr in enumerate(solver.qpos_addrs):
            data.qpos[addr] = q_ik[i]
    mujoco.mj_forward(model, data)


def read_arm_qpos_deg(data, ik):
    out = {}
    for side in ('left', 'right'):
        solver = ik.left if side == 'left' else ik.right
        q = np.array([data.qpos[a] for a in solver.qpos_addrs])
        out[side] = np.rad2deg(q).tolist()
    return out


def _ee_residual(data, ik, left_target, right_target):
    l = float(np.linalg.norm(data.site_xpos[ik.left.ee_site_id] - left_target[0]))
    r = float(np.linalg.norm(data.site_xpos[ik.right.ee_site_id] - right_target[0]))
    return l, r


def precompute(frames, model, data, ik):
    first_l = (frames[0][1], frames[0][2])
    first_r = (frames[0][3], frames[0][4])
    logger.info(f"  warmup: iterating toward first frame "
                f"(max {WARMUP_MAX_ITERS}, early-exit when stalled) ...")

    best_sum = float('inf')
    stall = 0
    iters = 0
    warmup_failures = 0
    with np.errstate(all='ignore'):
        for k in range(WARMUP_MAX_ITERS):
            try:
                step_ik_once(model, data, ik, first_l, first_r)
            except IKSolveError as e:
                warmup_failures += 1
                logger.warning(f"  warmup iter {k}: IK failed ({e}); continuing")
                continue
            iters = k + 1
            l_err, r_err = _ee_residual(data, ik, first_l, first_r)
            total = (l_err + r_err) * 1000.0
            if total < best_sum - WARMUP_TOL_MM:
                best_sum = total
                stall = 0
            else:
                stall += 1
            if stall >= WARMUP_STALL_ITERS:
                break
    l_err, r_err = _ee_residual(data, ik, first_l, first_r)
    logger.info(f"  warmup: {iters} iters (failures {warmup_failures}), residual "
                f"left={l_err*1000:.1f}mm right={r_err*1000:.1f}mm")

    joint_traj = []
    max_l = 0.0
    max_r = 0.0
    # Snapshot the current (post-warmup) joints as the fallback if a frame's
    # IK solve raises — we just hold at the previous known-good pose.
    last_good_q = read_arm_qpos_deg(data, ik)
    solve_failures = 0
    with np.errstate(all='ignore'):
        for i, (t, lp, lq, rp, rq) in enumerate(frames):
            try:
                step_ik_once(model, data, ik, (lp, lq), (rp, rq))
                last_good_q = read_arm_qpos_deg(data, ik)
            except IKSolveError as e:
                solve_failures += 1
                if solve_failures <= 5 or solve_failures % 50 == 0:
                    logger.warning(
                        f"  frame {i} (t={t:.4f}): IK failed ({e}); "
                        f"holding previous joints"
                    )
                # Restore data.qpos to the last good pose so the next frame's
                # sync_configuration reads a sane starting point.
                ja_rad = np.deg2rad(last_good_q['left'])
                jb_rad = np.deg2rad(last_good_q['right'])
                for j, addr in enumerate(ik.left.qpos_addrs):
                    data.qpos[addr] = ja_rad[j]
                for j, addr in enumerate(ik.right.qpos_addrs):
                    data.qpos[addr] = jb_rad[j]
                mujoco.mj_forward(model, data)
            l_err = float(np.linalg.norm(data.site_xpos[ik.left.ee_site_id] - lp))
            r_err = float(np.linalg.norm(data.site_xpos[ik.right.ee_site_id] - rp))
            if l_err > max_l: max_l = l_err
            if r_err > max_r: max_r = r_err
            joint_traj.append((t, list(last_good_q['left']), list(last_good_q['right'])))
    if solve_failures:
        logger.warning(f"  {solve_failures}/{len(frames)} frames used held-pose fallback")
    logger.info(f"  playback max EE tracking err: "
                f"left={max_l*1000:.1f}mm right={max_r*1000:.1f}mm")
    return joint_traj


# ---------------------------------------------------------------------------
# Collision check: arm/hand <-> prism, left side <-> right side, and each side
# against itself (with parent/grandparent exclusions so adjacent links don't
# false-positive).
# ---------------------------------------------------------------------------

# Substring-level body-name filter: the two robot subtrees plus the palm
# and finger bodies underneath each arm's link7.
_ARM_HAND_BODY_KEYWORDS = ('link', 'finger', 'palm', 'tip')
# Things that look like arm/hand bodies but aren't.
_SKIP_BODY_KEYWORDS = ('base_link', 'worldbody')


def _is_arm_hand_body(name):
    if not name:
        return False
    if any(k in name for k in _SKIP_BODY_KEYWORDS):
        return False
    return any(k in name for k in _ARM_HAND_BODY_KEYWORDS)


def _side_of_body(name):
    if name is None:
        return None
    if name.startswith('left_'):
        return 'left'
    if name.startswith('right_'):
        return 'right'
    return None


def _is_hand_body(name):
    """palm or finger body."""
    return 'palm' in name or 'finger' in name


def _is_upper_arm_link(name):
    """Upper-arm / elbow link (link1..link4). Excludes link5/6/7 which are
    wrist-adjacent and always sit close to the palm, and excludes palm/fingers.
    """
    if _is_hand_body(name):
        return False
    for suffix in ('link5', 'link6', 'link7'):
        if suffix in name:
            return False
    return 'link' in name  # catches left_link1..left_link4, right_link1..right_link4


def _gather_body_geom_map(model):
    """
    Return {body_id: (body_name, side, [geom_id, ...])} for every arm/hand
    body that owns at least one geom. `side` is 'left' or 'right'.
    """
    out = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not _is_arm_hand_body(name):
            continue
        side = _side_of_body(name)
        if side is None:
            continue
        n_geoms = int(model.body_geomnum[bid])
        if n_geoms == 0:
            continue
        adr = int(model.body_geomadr[bid])
        gids = list(range(adr, adr + n_geoms))
        out[bid] = (name, side, gids)
    return out


def _is_ancestor_within(model, bid_a, bid_b, hops):
    """True if bid_a is within `hops` parent links of bid_b (or vice versa)."""
    for _ in range(hops):
        if bid_a == bid_b:
            return True
        parent = int(model.body_parentid[bid_b])
        if parent == bid_b:  # reached world
            break
        bid_b = parent
    return bid_a == bid_b


def _body_pair_excluded(model, bid_a, bid_b, same_body_ok=False):
    """
    Skip pairs that are the same body, or within 2 hops of each other in the
    kinematic tree (parent, grandparent, sibling). 2 hops is a broad but
    effective mask for meshes that always overlap at joints.
    """
    if bid_a == bid_b:
        return not same_body_ok
    if _is_ancestor_within(model, bid_a, bid_b, hops=2):
        return True
    if _is_ancestor_within(model, bid_b, bid_a, hops=2):
        return True
    return False


def check_trajectory_collisions(model, data, ik, joint_traj,
                                tolerance_prism=TOLERANCE_PRISM_M,
                                tolerance_other=TOLERANCE_OTHER_M):
    """
    For every frame in `joint_traj`, push the joints into `data.qpos`, run
    mj_forward, and check:
      (A) arm/hand geoms <-> rectangular_prism_geom         (tolerance_prism)
      (B) left arm/hand <-> right arm/hand (all body pairs) (tolerance_other)
      (C) same-side upper-arm (link1..4) <-> own hand       (tolerance_other)
    A "hit" is a pair whose minimum inter-geom distance is below the relevant
    tolerance.

    Returns a list of (frame_idx, t_csv, label, distance) for each hit.
    Uses mujoco.mj_geomDistance which ignores contype/conaffinity and treats
    mesh geoms as their convex hulls.
    """
    body_map = _gather_body_geom_map(model)
    if not body_map:
        logger.warning("  collision check: no arm/hand bodies with geoms found")
        return []

    prism_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'rectangular_prism_geom')

    # Pre-build the body pairs we'll check every frame.
    #
    # Cross-side (LEFT vs RIGHT): every admissible body pair — hands and arms
    # of opposite sides should never touch.
    # Same-side: only arm_link[1..6] vs hand (palm / fingers). This catches
    # the "wrist folds into the forearm" case but skips hand-vs-hand and
    # hand-vs-hand-mount noise, and skips arm-vs-arm (adjacent links always
    # overlap at the joints).
    bids = sorted(body_map.keys())
    body_pairs = []  # list of (bid_a, bid_b, label, name_a, name_b)
    for i, ba in enumerate(bids):
        name_a, side_a, _ = body_map[ba]
        for bb in bids[i + 1:]:
            name_b, side_b, _ = body_map[bb]
            if _body_pair_excluded(model, ba, bb):
                continue
            if side_a == side_b:
                # Same arm: only keep upper-arm (link1..link4) ↔ hand pairs.
                # link5/6/7 are wrist-adjacent and always near the palm, so
                # including them just produces noise at 5 cm tolerance.
                a_arm = _is_upper_arm_link(name_a)
                a_hand = _is_hand_body(name_a)
                b_arm = _is_upper_arm_link(name_b)
                b_hand = _is_hand_body(name_b)
                if (a_arm and b_hand) or (b_arm and a_hand):
                    body_pairs.append((ba, bb, f'self-{side_a}', name_a, name_b))
                # else: hand↔hand, link5/6/7↔hand, arm↔arm — skipped
            else:
                body_pairs.append((ba, bb, 'LR', name_a, name_b))

    n_geoms_total = sum(len(v[2]) for v in body_map.values())
    logger.info(f"  checking {len(joint_traj)} frames: "
                f"{n_geoms_total} arm/hand geoms across {len(body_map)} bodies, "
                f"{len(body_pairs)} body pairs + prism, "
                f"tolerance prism={tolerance_prism*1000:.0f}mm other={tolerance_other*1000:.0f}mm")

    left_addrs = list(ik.left.qpos_addrs)
    right_addrs = list(ik.right.qpos_addrs)
    fromto = np.zeros(6)
    distmax = max(tolerance_prism, tolerance_other) + 0.02

    hits = []
    first_reported = 0

    def _min_dist_between_bodies(ba, bb):
        _, _, gids_a = body_map[ba]
        _, _, gids_b = body_map[bb]
        dmin = float('inf')
        for ga in gids_a:
            for gb in gids_b:
                d = mujoco.mj_geomDistance(model, data, ga, gb, distmax, fromto)
                if d < dmin:
                    dmin = d
                    if dmin <= 0.0:
                        return dmin
        return dmin

    def _min_dist_body_vs_geom(bid, gid):
        _, _, gids = body_map[bid]
        dmin = float('inf')
        for g in gids:
            d = mujoco.mj_geomDistance(model, data, g, gid, distmax, fromto)
            if d < dmin:
                dmin = d
                if dmin <= 0.0:
                    return dmin
        return dmin

    for i, (t_csv, ja_deg, jb_deg) in enumerate(joint_traj):
        ja_rad = np.deg2rad(ja_deg)
        jb_rad = np.deg2rad(jb_deg)
        for j, addr in enumerate(left_addrs):
            data.qpos[addr] = ja_rad[j]
        for j, addr in enumerate(right_addrs):
            data.qpos[addr] = jb_rad[j]
        mujoco.mj_forward(model, data)

        # (A) prism vs every arm/hand body
        if prism_gid >= 0:
            for bid, (bname, _, _) in body_map.items():
                d = _min_dist_body_vs_geom(bid, prism_gid)
                if d < tolerance_prism:
                    hits.append((i, t_csv, f"prism <-> {bname}", d))
                    if first_reported < 10:
                        logger.warning(f"  frame {i} t={t_csv:.4f}s  "
                                       f"prism <-> {bname}  dist={d*1000:.1f}mm")
                        first_reported += 1

        # (B) and (C): all admissible body pairs
        for ba, bb, kind, na, nb in body_pairs:
            d = _min_dist_between_bodies(ba, bb)
            if d < tolerance_other:
                hits.append((i, t_csv, f"{kind} {na} <-> {nb}", d))
                if first_reported < 10:
                    logger.warning(f"  frame {i} t={t_csv:.4f}s  "
                                   f"{kind} {na} <-> {nb}  dist={d*1000:.1f}mm")
                    first_reported += 1

    if hits and len(hits) > first_reported:
        logger.warning(f"  ... and {len(hits) - first_reported} more hits not shown")

    # Break hits down by category so prism hits don't hide cross-arm hits
    n_prism = sum(1 for h in hits if h[2].startswith('prism'))
    n_lr    = sum(1 for h in hits if h[2].startswith('LR'))
    n_self_l = sum(1 for h in hits if h[2].startswith('self-left'))
    n_self_r = sum(1 for h in hits if h[2].startswith('self-right'))
    logger.info(f"  hit breakdown: prism <-> arm/hand = {n_prism}, "
                f"left<->right = {n_lr}, "
                f"left arm<->own hand = {n_self_l}, "
                f"right arm<->own hand = {n_self_r}")
    return hits


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


def wait_for_convergence(robot, dcss, target_a, target_b, timeout=30.0, tol=0.3):
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

speed_factor = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SPEED_FACTOR
if speed_factor <= 0:
    raise ValueError(f"speed_factor must be > 0 (got {speed_factor})")

logger.info(f"Loading trajectory: {CSV_PATH}")
frames = load_trajectory(CSV_PATH)
duration = frames[-1][0] - frames[0][0]
rate = (len(frames) - 1) / duration if duration > 0 else 0.0
logger.info(f"  {len(frames)} synchronized frames, duration {duration:.2f}s, "
            f"avg rate {rate:.1f} Hz")
logger.info(f"  speed factor: {speed_factor}x "
            f"(playback duration {duration / speed_factor:.2f}s)")

logger.info(f"Loading MuJoCo scene: {SCENE_XML}")
model, data, ik = build_sim_and_ik()
logger.info(f"  scene loaded: nq={model.nq}, nu={model.nu}")
logger.info("  ik ready, sim seeded to ARM_DEFAULT_Q")

logger.info("Pre-computing joint trajectory (mink) ...")
joint_traj = precompute(frames, model, data, ik)
first_t, first_a, first_b = joint_traj[0]
last_t, last_a, last_b = joint_traj[-1]
logger.info(f"  first joints A (left):  {fmt(first_a)}")
logger.info(f"  first joints B (right): {fmt(first_b)}")
logger.info(f"  last  joints A (left):  {fmt(last_a)}")
logger.info(f"  last  joints B (right): {fmt(last_b)}")

# --- Collision safety check against the rectangular prism ---
if COLLISION_CHECK:
    logger.info("Checking for arm/hand <-> prism collisions ...")
    hits = check_trajectory_collisions(model, data, ik, joint_traj,
                                       tolerance_prism=TOLERANCE_PRISM_M,
                                       tolerance_other=TOLERANCE_OTHER_M)
    if hits:
        logger.error(
            f"COLLISION SAFETY: {len(hits)}/{len(joint_traj)} frames triggered "
            f"a collision hit (prism tol {TOLERANCE_PRISM_M*1000:.0f} mm, "
            f"other tol {TOLERANCE_OTHER_M*1000:.0f} mm). Aborting — no motion "
            f"will be sent to the robot."
        )
        logger.error(f"  first hit: frame {hits[0][0]} t={hits[0][1]:.4f}s "
                     f"{hits[0][2]} dist={hits[0][3]*1000:.1f}mm")
        sys.exit(1)
    logger.info(f"  clear — no prism/inter-arm/arm-hand hits")
else:
    logger.info("Collision check disabled (COLLISION_CHECK = False)")

# Upsample to STREAM_FREQ_HZ via linear joint-space interpolation between
# consecutive IK frames. Smooths out the 60 Hz stop-start jitter.
logger.info(f"Upsampling {len(joint_traj)} IK frames to {STREAM_FREQ_HZ} Hz ...")
_t_arr = np.array([jt[0] for jt in joint_traj])
_ja_arr = np.array([jt[1] for jt in joint_traj])  # (N, 7)
_jb_arr = np.array([jt[2] for jt in joint_traj])  # (N, 7)
_t_start_csv = _t_arr[0]
_t_end_csv = _t_arr[-1]
_n_stream = max(int((_t_end_csv - _t_start_csv) * STREAM_FREQ_HZ) + 1, 1)
_stream_ts = _t_start_csv + np.arange(_n_stream) * STREAM_PERIOD_S
# np.interp is per-column; build 7 interpolants for each arm
stream_a = np.column_stack([np.interp(_stream_ts, _t_arr, _ja_arr[:, j]) for j in range(7)])
stream_b = np.column_stack([np.interp(_stream_ts, _t_arr, _jb_arr[:, j]) for j in range(7)])
logger.info(f"  {_n_stream} streaming frames "
            f"({_stream_ts[-1] - _stream_ts[0]:.2f}s at {STREAM_FREQ_HZ}Hz)")

logger.info(f"Connecting to robot at {ROBOT_IP} ...")
robot, dcss = connect_robot()

robot.clear_set()
robot.clear_error('A')
robot.clear_error('B')
robot.send_cmd()
time.sleep(0.5)

robot.clear_set()
robot.set_state(arm='A', state=1)
robot.set_vel_acc(arm='A', velRatio=5, AccRatio=5)
robot.set_state(arm='B', state=1)
robot.set_vel_acc(arm='B', velRatio=5, AccRatio=5)
robot.send_cmd()
time.sleep(1.0)

logger.info("Moving both arms to start pose (slow) ...")
robot.clear_set()
robot.set_joint_cmd_pose(arm='A', joints=first_a)
robot.set_joint_cmd_pose(arm='B', joints=first_b)
robot.send_cmd()
time.sleep(0.2)
wait_for_convergence(robot, dcss, first_a, first_b, timeout=30, tol=0.3)
logger.info("  start pose reached")

robot.clear_set()
robot.set_vel_acc(arm='A', velRatio=50, AccRatio=50)
robot.set_vel_acc(arm='B', velRatio=50, AccRatio=50)
robot.send_cmd()
time.sleep(0.3)

# Seed sim qpos to frame 0 so the viewer shows the start pose before streaming.
_first_a_rad = np.deg2rad(stream_a[0])
_first_b_rad = np.deg2rad(stream_b[0])
for i, addr in enumerate(ik.left.qpos_addrs):
    data.qpos[addr] = _first_a_rad[i]
for i, addr in enumerate(ik.right.qpos_addrs):
    data.qpos[addr] = _first_b_rad[i]
mujoco.mj_forward(model, data)

logger.info(f"Streaming {_n_stream} frames at {STREAM_FREQ_HZ} Hz "
            f"(MuJoCo viewer enabled) ...")
late_frames = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.lookat[:] = [0.15, 0.0, 0.95]
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 180
    viewer.sync()

    t_start = time.monotonic()
    for k in range(_n_stream):
        if not viewer.is_running():
            logger.warning("viewer closed — aborting stream")
            break
        target_t = t_start + (_stream_ts[k] - _t_start_csv) / speed_factor
        now = time.monotonic()
        sleep_for = target_t - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            late_frames += 1

        robot.clear_set()
        robot.set_joint_cmd_pose(arm='A', joints=stream_a[k].tolist())
        robot.set_joint_cmd_pose(arm='B', joints=stream_b[k].tolist())
        robot.send_cmd()

        # Mirror the commanded joints in the sim + refresh the viewer at ~60 Hz
        if k % VIEWER_DIVISOR == 0:
            ja_rad = np.deg2rad(stream_a[k])
            jb_rad = np.deg2rad(stream_b[k])
            for i, addr in enumerate(ik.left.qpos_addrs):
                data.qpos[addr] = ja_rad[i]
            for i, addr in enumerate(ik.right.qpos_addrs):
                data.qpos[addr] = jb_rad[i]
            mujoco.mj_forward(model, data)
            viewer.sync()

    elapsed = time.monotonic() - t_start
    logger.info(f"  streaming done in {elapsed:.2f}s "
                f"(csv duration {_t_end_csv - _t_start_csv:.2f}s, "
                f"late frames: {late_frames}/{_n_stream})")

    # Keep the viewer up briefly at the final pose
    for _ in range(60):
        if not viewer.is_running():
            break
        viewer.sync()
        time.sleep(1.0 / 60.0)

wait_for_convergence(robot, dcss, last_a, last_b, timeout=10, tol=0.5)

logger.info("Disabling servos and releasing robot ...")
robot.clear_set()
robot.set_state(arm='A', state=0)
robot.set_state(arm='B', state=0)
robot.send_cmd()
time.sleep(0.5)
robot.release_robot()
logger.info("Done.")
