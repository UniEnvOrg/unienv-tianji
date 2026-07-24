"""
Rewrite DexTele's scene_tianji_wuji_full.xml with correct kinematics, using
the Fusion-exported Marvin URDF as the source of truth.

The sim_infra / calvinzqiu scene dropped every joint's <origin rpy=...> twist
and every theta-offset yaw, then tried to compensate by rewriting <joint axis>
in "sensible" directions. That compensation is wrong for joints 3, 4, 5, 7
(verified by diag_fk_consistency.py — FK delta ~660-770 mm vs. the manufacturer
SDK fx_kine at the current pose).

This script replaces the two arm body chains with a fresh one derived from
the Fusion URDF joint origins. Everything else in the scene (base_link mount,
Wuji hand subtree, meshes, actuators, collision exclusions, top-level options)
is preserved. The Wuji palm body is re-parented under the new link7 exactly
as it was attached before.

Fusion joint origins (LEFT and RIGHT are identical; mirroring comes from the
left/right base_link mount quaternions already in the scene):

    Joint1  xyz="0 0 0.1745"     rpy="0 0 0"
    Joint2  xyz="0 0 0"          rpy="1.5708 0 0"
    Joint3  xyz="0 0.287 0"      rpy="-1.5708 0 0"
    Joint4  xyz="0.018 0 0"      rpy="-1.5708 0 3.1416"
    Joint5  xyz="0.018 -0.314 0" rpy="-1.5708 0 3.1416"
    Joint6  xyz="0 0 0"          rpy="1.5708 -1.5708 0"
    Joint7  xyz="0 0 0"          rpy="1.5708 -1.5708 0"

All joint axes become "0 0 1" (local Z), matching the Fusion URDF and the
manufacturer MDH convention. Joint limits are copied verbatim from the
Fusion URDF (slightly wider than sim_infra's — J1/J3/J5 are ±3.1067 rad
instead of ±2.96706).
"""

import os
import math
import xml.etree.ElementTree as ET


def rpy_to_quat(roll, pitch, yaw):
    """URDF rpy (extrinsic XYZ = intrinsic ZYX) -> MJCF quat (w, x, y, z).

    URDF formula: R = R_z(yaw) * R_y(pitch) * R_x(roll)
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)

SRC = '/home/tianji/Desktop/tianji_teleop/DexTele/robots/tianji/scene_tianji_wuji_full.xml'
DST = '/home/tianji/Desktop/tianji_teleop/DexTele/robots/tianji/scene_tianji_wuji_full_fixed.xml'
DST_ARMS_ONLY = '/home/tianji/Desktop/tianji_teleop/DexTele/robots/tianji/scene_tianji_arms_only_fixed.xml'
DST_ARMS_WITH_HANDS = '/home/tianji/Desktop/tianji_teleop/DexTele/robots/tianji/scene_tianji_arms_with_hands_fixed.xml'

# (joint_number, xyz_str, rpy_str, range_str)  — from the Fusion URDFs
FUSION_JOINTS = [
    (1, "0 0 0.1745",      "0 0 0",               "-3.1067 3.1067"),
    (2, "0 0 0",            "1.5708 0 0",         "-2.0944 2.0944"),
    (3, "0 0.287 0",        "-1.5708 0 0",        "-3.1067 3.1067"),
    (4, "0.018 0 0",        "-1.5708 0 3.1416",   "-2.5307 1.0472"),
    (5, "0.018 -0.314 0",   "-1.5708 0 3.1416",   "-3.1067 3.1067"),
    (6, "0 0 0",            "1.5708 -1.5708 0",   "-1.0472 1.0472"),
    (7, "0 0 0",            "1.5708 -1.5708 0",   "-1.5708 1.5708"),
]

# Flange (TCP) static transform from link7's frame, from row 8 of the SDK MDH:
#   alpha=90 deg, a=0, d=95 mm, theta_offset=90 deg
# T_flange = Rot_x(pi/2) * Rot_z(pi/2) * Trans_z(0.095)
# In MJCF body form (Trans(pos) * Rot(quat)):
#   pos  = R * (0,0,0.095) = (0, -0.095, 0) in link7's local frame
#   quat = R = Rot_x(pi/2) * Rot_z(pi/2) = (0.5, 0.5, -0.5, 0.5)  (wxyz)
FLANGE_POS = "0 -0.095 0"
FLANGE_QUAT = "0.5 0.5 -0.5 0.5"

# Rectangular prism: 12 in x 12 in column under the robot, running from the
# floor up to the bottom of the OLD robot mount (the BASE_696 mesh that
# originally sat on `base_link` at z = 0.87). The mount mesh stays where it
# was; the prism is a new sibling body below it.
ARM_MOUNT_Z      = 1.01              # world z of left_base_link / right_base_link
MOUNT_BODY_Z     = 0.87              # world z of base_link (the BASE_696 mesh)
PRISM_TOP_Z      = MOUNT_BODY_Z      # prism reaches the bottom of the old mount

PRISM_HALF_XY    = 0.3048 / 2.0      # 0.1524 m
PRISM_HALF_Z     = PRISM_TOP_Z / 2.0 # bottom = 0, top = PRISM_TOP_Z

# Palm orientation in the flange's local frame. At all-zero joints we want
# the palm "front face" (palm-local -X) to point at world +X (forward) and the
# fingers (palm-local +Z) to point outward to the side (+Y for left arm,
# -Y for right arm). Both arms work with the same quat because the left/right
# base mounts differ by Rot_x(180 deg), which simultaneously flips the side
# direction the way we want.
#
# Quat (0, 0, 0, 1) = 180 deg rotation about local Z axis.
PALM_IN_FLANGE_QUAT = "0.0 0.0 0.0 1.0"

# Per-link mesh asset names in the existing scene
MESH_NAMES_L = ["Link1_L", "Link2_L", "Link3_L", "Link4_L", "Link5_L", "Link6_L", "Link7_L"]
MESH_NAMES_R = ["Link1_R", "Link2_R", "Link3_R", "Link4_R", "Link5_R", "Link6_R", "Link7_R"]


def find_body(root, name):
    for b in root.iter('body'):
        if b.get('name') == name:
            return b
    return None


def parent_map(root):
    return {c: p for p in root.iter() for c in p}


def rewrite_arm(worldbody, side):
    base = find_body(worldbody, f'{side}_base_link')
    if base is None:
        raise RuntimeError(f"missing {side}_base_link")

    # --- Find the existing link1 (child of base_link) and the palm subtree ---
    old_link1 = None
    for c in list(base):
        if c.tag == 'body' and c.get('name') == f'{side}_link1':
            old_link1 = c
            break
    if old_link1 is None:
        raise RuntimeError(f"missing {side}_link1 under {side}_base_link")

    # Walk the existing chain to find link7 and then the palm body
    old_link7 = find_body(base, f'{side}_link7')
    palm_body = None
    site_ee = None
    if old_link7 is not None:
        for c in list(old_link7):
            if c.tag == 'body' and 'palm' in (c.get('name') or ''):
                palm_body = c
            if c.tag == 'site' and c.get('name') == f'{side}_ee_site':
                site_ee = c

    # --- Remove the old chain entirely (keeps base_link's base mesh geom etc.) ---
    base.remove(old_link1)

    # --- Build the new chain ---
    mesh_names = MESH_NAMES_L if side == 'left' else MESH_NAMES_R
    parent = base
    new_link_bodies = []
    for (jnum, xyz, rpy, rng), mesh in zip(FUSION_JOINTS, mesh_names):
        body = ET.SubElement(parent, 'body')
        body.set('name', f'{side}_link{jnum}')
        body.set('gravcomp', '1')
        if xyz != '0 0 0':
            body.set('pos', xyz)
        if rpy != '0 0 0':
            # URDF rpy uses extrinsic XYZ = R_z(yaw)*R_y(pitch)*R_x(roll).
            # MJCF default euler is intrinsic XYZ which is a different
            # composition — convert to a quaternion to be unambiguous.
            r, p, y = (float(v) for v in rpy.split())
            qw, qx, qy, qz = rpy_to_quat(r, p, y)
            body.set('quat', f'{qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f}')

        joint = ET.SubElement(body, 'joint')
        joint.set('name', f'{side}_joint{jnum}')
        joint.set('pos', '0 0 0')
        joint.set('axis', '0 0 1')  # all joints rotate about local Z in MDH
        joint.set('range', rng)

        # Visual mesh — Fusion URDF uses these same STLs with NO per-geom quat
        geom = ET.SubElement(body, 'geom')
        geom.set('type', 'mesh')
        geom.set('mesh', mesh)
        geom.set('rgba', '0.8 0.8 0.8 1')
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
        geom.set('group', '1')

        new_link_bodies.append(body)
        parent = body

    # Add a static "flange" body on link7 for the SDK MDH row 8 transform.
    # The ee_site and the Wuji palm both live INSIDE the flange so their
    # poses match the manufacturer SDK's TCP convention exactly.
    link7 = new_link_bodies[-1]
    flange = ET.SubElement(link7, 'body')
    flange.set('name', f'{side}_flange')
    flange.set('pos', FLANGE_POS)
    flange.set('quat', FLANGE_QUAT)

    if site_ee is not None:
        # Re-anchor ee_site at the flange origin (pos=0,0,0) so its world
        # pose IS the SDK's reported TCP pose.
        site_ee.set('pos', '0 0 0')
        if 'quat' in site_ee.attrib:
            del site_ee.attrib['quat']
        flange.append(site_ee)
    if palm_body is not None:
        # Move the Wuji palm to the flange. The original pos/quat were baked
        # for the broken link7 frame; replace them so the palm sits at the
        # flange origin with PALM_IN_FLANGE_QUAT — this orients the hand so
        # its fingers (palm local +Z) point at world +X at all-zero joints.
        palm_body.set('pos', '0 0 0')
        palm_body.set('quat', PALM_IN_FLANGE_QUAT)
        flange.append(palm_body)


def build_arms_only_scene():
    """
    Build a minimal MJCF scene from scratch containing only the two Tianji
    arms with the corrected kinematics + flange. No Wuji hand, no torso skin,
    no table, no objects, no contact exclusions, no auxiliary lighting beyond
    the bare minimum needed for the viewer.
    """
    mj = ET.Element('mujoco', {'model': 'tianji_arms_only_fixed'})

    ET.SubElement(mj, 'compiler', {'angle': 'radian', 'meshdir': '../../assets/tianji_arm_ccs696/meshes'})
    ET.SubElement(mj, 'option', {'timestep': '0.002', 'integrator': 'implicitfast'})

    visual = ET.SubElement(mj, 'visual')
    ET.SubElement(visual, 'global', {'azimuth': '180', 'elevation': '-25'})
    ET.SubElement(visual, 'quality', {'shadowsize': '2048'})

    asset = ET.SubElement(mj, 'asset')
    ET.SubElement(asset, 'texture', {
        'type': 'skybox', 'builtin': 'gradient', 'rgb1': '0.3 0.5 0.7',
        'rgb2': '0 0 0', 'width': '512', 'height': '512',
    })
    ET.SubElement(asset, 'texture', {
        'name': 'groundplane', 'type': '2d', 'builtin': 'checker',
        'mark': 'edge', 'rgb1': '0.2 0.3 0.4', 'rgb2': '0.1 0.2 0.3',
        'markrgb': '0.8 0.8 0.8', 'width': '300', 'height': '300',
    })
    ET.SubElement(asset, 'material', {
        'name': 'groundplane', 'texture': 'groundplane',
        'texuniform': 'true', 'texrepeat': '5 5', 'reflectance': '0.2',
    })
    for side in ('L', 'R'):
        for name in [f'Base_{side}'] + [f'Link{i}_{side}' for i in range(1, 8)]:
            sub = 'left_arm' if side == 'L' else 'right_arm'
            ET.SubElement(asset, 'mesh', {
                'name': name,
                'file': f'{sub}/{name}.STL',
            })
    # Original DexTele mount mesh (BASE_696): the top plate the robot bolts to
    ET.SubElement(asset, 'mesh', {
        'name': 'BASE_696',
        'file': 'SRS/BASE_696.stl',
        'scale': '0.001 0.001 0.001',
    })

    wb = ET.SubElement(mj, 'worldbody')
    ET.SubElement(wb, 'light', {'pos': '0.5 0 2.5', 'dir': '0 0 -1', 'directional': 'true'})
    ET.SubElement(wb, 'geom', {
        'name': 'floor', 'pos': '0 0 0', 'size': '0 0 0.05',
        'type': 'plane', 'material': 'groundplane',
    })

    # Rectangular prism (the column under the old DexTele mount): 12x12 in
    # cross-section, from floor to the bottom of the mount body.
    prism = ET.SubElement(wb, 'body', {
        'name': 'rectangular_prism',
        'pos': f'0 0 {PRISM_HALF_Z}',
    })
    ET.SubElement(prism, 'geom', {
        'name': 'rectangular_prism_geom',
        'type': 'box',
        'size': f'{PRISM_HALF_XY} {PRISM_HALF_XY} {PRISM_HALF_Z}',
        'rgba': '0.45 0.45 0.5 1',
        'contype': '1',
        'conaffinity': '1',
        'group': '1',
    })
    # Old DexTele mount mesh on top of the prism
    base_link_body = ET.SubElement(wb, 'body', {
        'name': 'base_link',
        'pos': f'0 0 {MOUNT_BODY_Z}',
    })
    ET.SubElement(base_link_body, 'geom', {
        'type': 'mesh',
        'mesh': 'BASE_696',
        'rgba': '1 1 1 1',
        'contype': '0',
        'conaffinity': '0',
        'group': '1',
    })

    arm_specs = [
        ('left',  f'0.0  0.04 {ARM_MOUNT_Z}', '0.707105 -0.707108 0 0', 'L'),
        ('right', f'0.0 -0.04 {ARM_MOUNT_Z}', '0.707105  0.707108 0 0', 'R'),
    ]
    for side, base_pos, base_quat, suffix in arm_specs:
        arm_base = ET.SubElement(wb, 'body', {
            'name': f'{side}_base_link',
            'pos': base_pos,
            'quat': base_quat,
        })
        ET.SubElement(arm_base, 'geom', {
            'type': 'mesh', 'mesh': f'Base_{suffix}',
            'rgba': '0.8 0.8 0.8 1',
            'contype': '0', 'conaffinity': '0', 'group': '1',
        })

        parent = arm_base
        link7_body = None
        for jnum, xyz, rpy, rng in FUSION_JOINTS:
            body = ET.SubElement(parent, 'body', {'name': f'{side}_link{jnum}'})
            if xyz != '0 0 0':
                body.set('pos', xyz)
            if rpy != '0 0 0':
                r, p, y = (float(v) for v in rpy.split())
                qw, qx, qy, qz = rpy_to_quat(r, p, y)
                body.set('quat', f'{qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f}')
            ET.SubElement(body, 'joint', {
                'name': f'{side}_joint{jnum}',
                'pos': '0 0 0', 'axis': '0 0 1', 'range': rng,
            })
            ET.SubElement(body, 'geom', {
                'type': 'mesh', 'mesh': f'Link{jnum}_{suffix}',
                'rgba': '0.8 0.8 0.8 1',
                'contype': '0', 'conaffinity': '0', 'group': '1',
            })
            parent = body
            if jnum == 7:
                link7_body = body

        # Static flange + ee_site (matches SDK MDH row 8)
        flange = ET.SubElement(link7_body, 'body', {
            'name': f'{side}_flange',
            'pos': FLANGE_POS, 'quat': FLANGE_QUAT,
        })
        ET.SubElement(flange, 'site', {
            'name': f'{side}_ee_site',
            'pos': '0 0 0', 'size': '0.01',
            'rgba': '1 0 0 0.5',
        })

    # Position actuators on each arm joint so mink can drive them
    actuator = ET.SubElement(mj, 'actuator')
    for side in ('left', 'right'):
        for jnum, _, _, rng in FUSION_JOINTS:
            ET.SubElement(actuator, 'position', {
                'name': f'{side}_joint{jnum}_actuator',
                'joint': f'{side}_joint{jnum}',
                'kp': '500',
                'ctrlrange': rng,
            })

    tree = ET.ElementTree(mj)
    ET.indent(tree, space='  ', level=0)
    tree.write(DST_ARMS_ONLY, xml_declaration=True, encoding='utf-8')
    print(f"wrote {DST_ARMS_ONLY}")


def build_arms_with_hands_scene():
    """
    Take the full fixed scene and strip every non-arm body from worldbody
    (table, objects, bins, pillar, pole_mount). Keep base_link, both arms
    (which include the Wuji palm + finger subtree), lights, and floor.
    Also clean up <contact> exclude pairs that reference removed bodies.
    Mesh assets for the removed bodies are left alone — unused mesh
    declarations are harmless and keep the diff small.
    """
    KEEP_BODIES = {'base_link', 'left_base_link', 'right_base_link'}

    tree = ET.parse(DST)
    root = tree.getroot()
    wb = root.find('worldbody')

    removed_body_names = set()
    removed_joint_names = set()
    for child in list(wb):
        if child.tag != 'body':
            continue
        name = child.get('name')
        if name in KEEP_BODIES:
            continue
        # Collect all body and joint names under this subtree so we can
        # prune any contact / sensor / actuator entries that reference them.
        for sub in child.iter('body'):
            sn = sub.get('name')
            if sn:
                removed_body_names.add(sn)
        for jn in child.iter('joint'):
            n = jn.get('name')
            if n:
                removed_joint_names.add(n)
        for fj in child.iter('freejoint'):
            n = fj.get('name')
            if n:
                removed_joint_names.add(n)
        wb.remove(child)

    # Prune <contact> exclude pairs that reference any removed body
    contact = root.find('contact')
    if contact is not None:
        for excl in list(contact):
            if excl.tag != 'exclude':
                continue
            b1 = excl.get('body1', '')
            b2 = excl.get('body2', '')
            if b1 in removed_body_names or b2 in removed_body_names:
                contact.remove(excl)

    # Prune <sensor> entries that reference removed bodies / joints
    sensor = root.find('sensor')
    if sensor is not None:
        for s in list(sensor):
            ref = s.get('joint') or s.get('body') or s.get('site') or ''
            if ref in removed_body_names or ref in removed_joint_names:
                sensor.remove(s)

    # Prune <actuator> entries that drive removed joints
    actuator = root.find('actuator')
    if actuator is not None:
        for a in list(actuator):
            j = a.get('joint', '')
            if j in removed_joint_names:
                actuator.remove(a)

    # Prune <keyframe> entries — they typically encode qpos for the full
    # model and their length will be wrong after stripping bodies. Just drop
    # them; the arm+hand scene doesn't need them.
    for kf in list(root.findall('keyframe')):
        root.remove(kf)

    # Leave the original DexTele `base_link` body alone — it carries the
    # BASE_696 mount mesh at z = MOUNT_BODY_Z (0.87). Just add a new sibling
    # rectangular-prism body so the mount visually sits on top of a column.
    prism = ET.SubElement(wb, 'body', {
        'name': 'rectangular_prism',
        'pos': f'0 0 {PRISM_HALF_Z}',
    })
    ET.SubElement(prism, 'geom', {
        'name': 'rectangular_prism_geom',
        'type': 'box',
        'size': f'{PRISM_HALF_XY} {PRISM_HALF_XY} {PRISM_HALF_Z}',
        'rgba': '0.45 0.45 0.5 1',
        'contype': '1',
        'conaffinity': '1',
        'group': '1',
    })

    # left_base_link / right_base_link keep their original z (1.01) from the
    # full-fixed source — no repositioning needed.

    ET.indent(tree, space='  ', level=0)
    tree.write(DST_ARMS_WITH_HANDS, xml_declaration=True, encoding='utf-8')
    print(f"wrote {DST_ARMS_WITH_HANDS}  (removed {len(removed_body_names)} bodies, "
          f"{len(removed_joint_names)} joints, pedestal added)")


def main():
    ET.register_namespace('', '')  # keep tags unprefixed
    tree = ET.parse(SRC)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError("no worldbody in source XML")

    rewrite_arm(worldbody, 'left')
    rewrite_arm(worldbody, 'right')

    # Pretty-indent for readability (Python 3.9+)
    ET.indent(tree, space="  ", level=0)
    tree.write(DST, xml_declaration=True, encoding='utf-8')
    print(f"wrote {DST}")

    build_arms_only_scene()
    build_arms_with_hands_scene()


if __name__ == '__main__':
    main()
