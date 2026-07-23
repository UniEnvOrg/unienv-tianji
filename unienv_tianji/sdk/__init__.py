# Vendored Tianji/Marvin robot arm control SDK.
#
# This subpackage contains a verbatim copy of the Python SDK from
# https://github.com/calvinzqiu/tianji_teleop (tianji-arm, Apache-2.0,
# Copyright 2025 上海孚晞科技有限公司). It is vendored here so that the
# unienv_tianji package is self-contained.
#
# The SDK loads native shared libraries (libMarvinSDK.so / libKine.so on Linux,
# libMarvinSDK.dll / libKine.dll on Windows) from this same directory (the path
# is resolved relative to fx_robot.py / fx_kine.py via __file__). Users must
# place those binaries next to the vendored files (i.e. inside this
# unienv_tianji/sdk/ directory) or build them from the upstream
# tianji-arm/contrlSDK and tianji-arm/kinematicsSDK sources.
