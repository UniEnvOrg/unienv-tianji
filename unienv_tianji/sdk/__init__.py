# Vendored Tianji/Marvin robot arm control SDK.
#
# This subpackage contains a verbatim copy of the Python SDK from
# https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK (branch: master,
# subdirectory: SDK_PYTHON, Apache-2.0, Copyright 2025 上海孚晞科技有限公司).
# It is vendored here so that the unienv_tianji package is self-contained.
#
# The SDK loads native shared libraries (libMarvinSDK.so / libKine.so on Linux,
# libMarvinSDK.dll / libKine.dll on Windows) from this same directory (the path
# is resolved relative to fx_robot.py / fx_kine.py via __file__). The official
# prebuilt binaries for Linux x86_64 and Windows ship inside this directory and
# are redistributed under the same Apache-2.0 license. Users on other platforms
# can build them from the upstream repo's marvinSDK_ubuntu.sh /
# marvinSDK_windows.bat scripts.