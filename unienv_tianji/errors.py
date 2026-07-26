"""Hardware error type for Tianji/Marvin arm actors.

Raised by :class:`unienv_tianji.TianjiArmActor` (and the EEF node wrapping it)
when a feedback refresh reports a non-zero ``err_code`` or an error
``cur_state`` (100), signalling that the arm has tripped a fault and needs
:py:meth:`~unienv_tianji.TianjiArmActor.clear_errors` before it can be driven
again.
"""

from typing import Optional


class TianjiArmHardwareError(RuntimeError):
    """The arm reported a hardware fault on its feedback channel.

    Carries the offending arm id (``"A"`` / ``"B"``), the current state code
    (``cur_state``) and the vendor error code (``err_code``) so callers can log
    diagnostics without re-querying the controller.

    Recovery is via :py:meth:`TianjiArmActor.clear_errors
    <unienv_tianji.tianji_arm.TianjiArmActor.clear_errors>` followed by
    re-enabling the arm.
    """

    def __init__(
        self,
        arm: str,
        cur_state: Optional[int] = None,
        error_code: Optional[int] = None,
        message: Optional[str] = None,
    ):
        self.arm = arm
        self.cur_state = cur_state
        self.error_code = error_code
        if message is None:
            message = (
                f"Tianji arm {arm!r} reported a hardware fault "
                f"(cur_state={cur_state}, err_code={error_code}). "
                f"Call clear_errors() on the actor to recover, then re-enable the arm."
            )
        super().__init__(message)