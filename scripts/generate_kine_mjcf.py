#!/usr/bin/env python
"""Generate kinematics-only per-arm MJCFs from the dual-arm source MJCF.

The source model at
``genesis_adaptor/unienv_genesis_collection/assets/robots/tianji_marvin_ccs_696/tianji_marvin_CCS_real_limit_wuji.xml``
describes the full Tianji/Marvin dual-arm robot with Wuji hands (meshes,
geoms, finger joints, actuators, sensors, contacts). For pure kinematics +
mink-based IK we only need one 7-DoF arm chain per file, welded to a palm
frame, with no meshes/geoms/actuators/sensors/finger-joints.

This script derives, for each arm (``A``=left / ``B``=right), a self-contained
MJCF that:

* keeps the source ``<compiler>``/``<option>`` (minus anything MuJoCo
  rejects), the joint defaults for the 7 arm joints, the 7 arm bodies with
  their ``<inertial>`` and ``<joint>`` elements, and the palm body welded to
  Link7 (no finger joints);
* drops the *other* arm's subtree entirely;
* removes all ``<mesh>`` assets, ``<material>`` assets, ``<geom>`` elements,
  ``<actuator>``s, ``<contact>`` excludes and the keyframe ctrl values;
* adds a ``<site name="palm">`` on the palm body replicating the source palm
  body frame (pos + xyaxes), so the IK target frame is unambiguous;
* adds a ``home`` keyframe holding the arm's home qpos (radians).

The generated files are what ship under ``unienv_tianji/assets/``; this
script documents the derivation and can be re-run if the source MJCF
changes.

Usage
-----
::

    python scripts/generate_kine_mjcf.py [SOURCE_XML] [OUT_DIR]

Defaults:
    SOURCE_XML = .../tianji_marvin_ccs_696/tianji_marvin_CCS_real_limit_wuji.xml
    OUT_DIR    = <repo>/unienv_tianji/assets
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# Home joint positions (radians), matching REST_JOINT_POSITIONS in
# unienv_tianji/tianji_arm.py (left arm j5 sign flipped vs. the source
# keyframe).
HOME_QPOS = {
    "A": [0.2, -0.963, 0.0, -0.85, 0.356, 0.0, 0.0],   # left
    "B": [-0.2, -0.963, 0.0, -0.85, -0.356, 0.0, 0.0],  # right
}

# Per-arm root link body name + root link pos/quat taken verbatim from the
# source MJCF's worldbody (so the per-arm model's world frame == the source
# robot root frame used directly by EEF actions and observations for both arms).
ARM_ROOT = {
    "A": {
        "body": "Link1_L",
        "pos": "0 0.2145 0.139999",
        "quat": "0.707105 -0.707108 0 0",
    },
    "B": {
        "body": "Link1_R",
        "pos": "0 -0.2145 0.139999",
        "quat": "0.707105 0.707108 0 0",
    },
}

# Per-arm chain of (body_name, joint_name) from the source MJCF.
ARM_CHAIN = {
    "A": [
        ("Link1_L", "Joint1_L"),
        ("Link2_L", "Joint2_L"),
        ("Link3_L", "Joint3_L"),
        ("Link4_L", "Joint4_L"),
        ("Link5_L", "Joint5_L"),
        ("Link6_L", "Joint6_L"),
        ("Link7_L", "Joint7_L"),
    ],
    "B": [
        ("Link1_R", "Joint1_R"),
        ("Link2_R", "Joint2_R"),
        ("Link3_R", "Joint3_R"),
        ("Link4_R", "Joint4_R"),
        ("Link5_R", "Joint5_R"),
        ("Link6_R", "Joint6_R"),
        ("Link7_R", "Joint7_R"),
    ],
}

PALM_BODY = {"A": "left_wuji_palm", "B": "right_wuji_palm"}
# Palm body frame (pos + xyaxes) from the source MJCF (lines 131 / 301).
PALM_FRAME = {
    "A": {"pos": "0 -0.09 0", "xyaxes": "-0.707107 0 0.707107 -0.707107 0 -0.707107"},
    "B": {"pos": "0 -0.09 0", "xyaxes": "0.707107 0 0.707107 -0.707107 0 0.707107"},
}

# Default class names used by the 7 arm joints in the source MJCF.
ARM_DEFAULT_CLASSES = {"tianji_arm", "tianji_wrist5", "tianji_wrist6", "tianji_wrist7"}


def _copy_inertial(src_body: ET.Element, dst_body: ET.Element) -> None:
    """Copy the <inertial> child (if any) verbatim."""
    inertial = src_body.find("inertial")
    if inertial is not None:
        dst_body.append(_clone(inertial))


def _copy_joint(src_body: ET.Element, dst_body: ET.Element) -> None:
    """Copy the <joint> child verbatim (only one per arm body in the chain)."""
    joint = src_body.find("joint")
    if joint is not None:
        dst_body.append(_clone(joint))


def _clone(el: ET.Element) -> ET.Element:
    """Deep-clone an ElementTree element, detaching it from its parent."""
    return ET.fromstring(ET.tostring(el))


def _strip_attrs(el: ET.Element, attrs: set[str]) -> None:
    for a in attrs:
        if a in el.attrib:
            del el.attrib[a]


def build_arm_xml(arm: str, source_root: ET.Element) -> ET.Element:
    """Build a kinematics-only per-arm MJCF Element from the source root."""
    src_compiler = source_root.find("compiler")
    src_option = source_root.find("option")
    src_default = source_root.find("default")
    src_worldbody = source_root.find("worldbody")

    # ---- New <mujoco> root ----
    new_root = ET.Element("mujoco", {"model": f"tianji_{arm}_arm_kine"})
    # compiler: keep angle + a relative meshdir-less config. Drop meshdir
    # (no meshes) and any other Genesis-specific attrs.
    compiler = ET.SubElement(new_root, "compiler")
    for k, v in (src_compiler.attrib if src_compiler is not None else {}).items():
        if k in ("meshdir", "texturedir", "autolimits"):
            continue
        compiler.set(k, v)
    if "angle" not in compiler.attrib:
        compiler.set("angle", "radian")
    if "coordinate" not in compiler.attrib:
        # MuJoCo default is local; keep explicit for clarity.
        compiler.set("coordinate", "local")

    # option: keep as-is (integrator/cone are harmless for FK-only use).
    if src_option is not None:
        new_root.append(_clone(src_option))

    # ---- <default>: keep only the arm-joint default classes used by the 7 joints.
    if src_default is not None:
        new_default = ET.SubElement(new_root, "default")
        for child in src_default:
            if child.tag == "default" and child.get("class") in ARM_DEFAULT_CLASSES:
                # Keep only the <joint> child of each class (drop <geom>).
                cls = ET.SubElement(new_default, "default", {"class": child.get("class")})
                for jc in child:
                    if jc.tag == "joint":
                        cls.append(_clone(jc))

    # ---- <worldbody> ----
    new_worldbody = ET.SubElement(new_root, "worldbody")
    # A single fixed "base" body (no joint) at the origin, mirroring the
    # source worldbody's "base" body's role but without geoms. The arm root
    # link hangs off it with the source root-link pos/quat so the per-arm
    # model's world frame == the source robot root frame.
    base = ET.SubElement(new_worldbody, "body", {"name": "base"})

    # Walk the chain: parent each next link under the previous one.
    parent_body = base
    src_bodies = {b.get("name"): b for b in src_worldbody.iter("body")}
    for i, (body_name, _joint_name) in enumerate(ARM_CHAIN[arm]):
        src_body = src_bodies[body_name]
        # Build the new body with pos/quat from the source.
        attribs = {"name": body_name}
        if "pos" in src_body.attrib:
            attribs["pos"] = src_body.attrib["pos"]
        if "quat" in src_body.attrib:
            attribs["quat"] = src_body.attrib["quat"]
        new_body = ET.SubElement(parent_body, "body", attribs)
        _copy_inertial(src_body, new_body)
        _copy_joint(src_body, new_body)
        parent_body = new_body

    # Palm body welded to Link7 (no joint): keep pos + xyaxes verbatim.
    palm_attribs = {"name": PALM_BODY[arm]}
    palm_attribs["pos"] = PALM_FRAME[arm]["pos"]
    palm_attribs["xyaxes"] = PALM_FRAME[arm]["xyaxes"]
    palm = ET.SubElement(parent_body, "body", palm_attribs)
    # A trivial inertial so MuJoCo is happy with a moving-descendant body
    # welded to a jointed parent (Link7). Tiny mass at the palm origin.
    ET.SubElement(
        palm, "inertial",
        {"pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-9 1e-9 1e-9"},
    )
    # palm site replicating the palm body frame (independent of welding).
    ET.SubElement(
        palm, "site",
        {"name": "palm", "pos": "0 0 0", "xyaxes": "1 0 0 0 1 0"},
    )

    # ---- <keyframe>: home qpos for this arm only (7 values). ----
    keyframe = ET.SubElement(new_root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": " ".join(f"{v:.6f}" for v in HOME_QPOS[arm])},
    )

    return new_root


def main(argv: list[str]) -> int:
    repo = Path(__file__).resolve().parent.parent
    default_source = repo.parent / (
        "genesis_adaptor/unienv_genesis_collection/assets/robots/"
        "tianji_marvin_ccs_696/tianji_marvin_CCS_real_limit_wuji.xml"
    )
    source_xml = Path(argv[1] if len(argv) > 1 else default_source)
    out_dir = Path(argv[2] if len(argv) > 2 else repo / "unienv_tianji" / "assets")
    if not source_xml.exists():
        print(f"error: source MJCF not found: {source_xml}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_xml)
    source_root = tree.getroot()

    # Pretty-printing.
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass  # Python < 3.9

    for arm, fname in (("A", "tianji_marvin_left_arm_kine.xml"),
                      ("B", "tianji_marvin_right_arm_kine.xml")):
        new_root = build_arm_xml(arm, source_root)
        new_tree = ET.ElementTree(new_root)
        try:
            ET.indent(new_tree, space="  ")
        except AttributeError:
            pass
        out_path = out_dir / fname
        new_tree.write(out_path, encoding="utf-8", xml_declaration=True)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
