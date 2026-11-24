# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Space metadata: `TianjiArmEefActor`'s `last_ik_solved_error` is described as mink's `FrameTask` error (the SE(3) log of `inv(achieved) @ target`) in the palm site frame (`frame="link_local"`, `site="palm"`); its `joint_torque` keeps the child's "joint-side sensor torque" label; the EEF action's `label` records the rot6d convention (first two rows of R, XBArray / pytorch3d) and, with `normalize_rotation_action`, that the Euler angles are divided by pi.

## 0.0.1a1 (2026-10-17)

- Initial public release
- Attach UniEnv space metadata to every observation and action space of `TianjiArmActor` and `TianjiArmEefActor` (units, joint names, frame, rotation representation, control / mode, source node and `fps`); requires `unienv>=0.0.1b16`.
- Licensed under the Apache License 2.0; `NOTICE` carries the copyright and third-party attributions.
