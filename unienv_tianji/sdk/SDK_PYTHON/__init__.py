# Python bindings for the vendored Tianji/Marvin SDK.
#
# This __init__.py is the ONLY file added to the otherwise verbatim SDK
# directory (the upstream typo'd marker file __ini__.py is left untouched).
# It makes fx_robot / fx_kine importable as a subpackage:
#     from unienv_tianji.sdk.SDK_PYTHON import fx_robot, fx_kine
