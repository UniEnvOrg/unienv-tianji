# Vendored Tianji/Marvin robot arm SDK (1003-generation).
#
# Verbatim copy of the tianji-arm SDK directory from
# https://github.com/calvinzqiu/tianji_teleop (subdirectory: tianji-arm,
# Apache-2.0, Copyright 2025 上海孚晞科技有限公司), itself derived from
# https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK. The original directory
# structure is preserved: Python bindings live in SDK_PYTHON/ (fx_robot.py,
# fx_kine.py), native sources in contrlSDK/ and kinematicsSDK/ (built via
# marvinSDK_ubuntu.sh into SDK_PYTHON/*.so), docs/demos/configs in place.
#
# We use this 1003-generation SDK (SDK_VERSION 1003 in contrlSDK/Robot.h)
# instead of the newer contrlSDK100343 because the target controller runs
# 1003-generation firmware; the 100343 SDK fails its VERSION handshake against
# it ("control system 0") and gates off all motion commands.
#
# Import the bindings as:
#     from unienv_tianji.sdk.SDK_PYTHON import fx_robot, fx_kine
