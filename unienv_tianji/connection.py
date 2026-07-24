"""Shared Tianji/Marvin controller connection.

The vendored SDK (``unienv_tianji.sdk.SDK_PYTHON.fx_robot``) only allows ONE controller
connection per process: it binds a fixed UDP port and exchanges data through a
single shared-memory ``DCSS`` structure, and ``clear_set()`` / ``send_cmd()``
operate on a process-global command buffer. To drive the dual-arm robot (arm
``"A"`` and arm ``"B"``) from a single process, both arms must therefore share
one :class:`TianjiConnection` instance and serialize command transactions.

This module wraps a single ``Marvin_Robot`` + ``DCSS`` pair behind a
re-entrant lock so that multiple :class:`~unienv_tianji.tianji_arm.TianjiArmActor`
instances can safely share it.
"""

import threading
import time
from contextlib import contextmanager
from typing import Dict, Generator

from .sdk.SDK_PYTHON import fx_robot


class TianjiConnection:
    """Shared, thread-safe handle to a single Tianji/Marvin controller.

    A single ``Marvin_Robot`` connection and ``DCSS`` subscription structure
    are owned by this object. All command transactions are serialized with a
    re-entrant lock and wrapped in a ``clear_set()`` / ``send_cmd()`` pair so
    that multi-call command sequences are atomic across threads.

    Parameters
    ----------
    ip:
        Controller IPv4 address (e.g. ``"192.168.1.190"``).
    connect:
        If ``True`` (default), connect to the controller immediately. If
        ``False``, the object is constructed disconnected and ``connect()``
        must be called explicitly before use.
    startup_settle:
        Seconds to sleep after ``robot.connect(ip)`` returns, before
        validating the link. Lets the controller's publish thread spin up.
    frame_check_count:
        Number of consecutive frames whose ``frame_serial`` must strictly
        advance for *both* arms before the link is considered healthy.
    frame_check_timeout:
        Maximum seconds to wait for ``frame_check_count`` advancing frames
        before giving up (best-effort ``release_robot()`` + ``ConnectionError``).
    """

    def __init__(
        self,
        ip: str,
        *,
        connect: bool = True,
        startup_settle: float = 0.5,
        frame_check_count: int = 10,
        frame_check_timeout: float = 2.0,
    ):
        self.ip = ip
        self._startup_settle = startup_settle
        self._frame_check_count = frame_check_count
        self._frame_check_timeout = frame_check_timeout

        self._lock = threading.RLock()
        self._robot = None
        self._dcss = None
        self._connected = False

        if connect:
            self.connect()

    # ========== Properties ==========
    @property
    def connected(self) -> bool:
        """Whether the controller link is currently established."""
        return self._connected

    # ========== Lifecycle ==========
    def connect(self) -> None:
        """Establish the controller link and validate it.

        Creates the ``Marvin_Robot`` / ``DCSS`` pair, calls
        ``robot.connect(ip)``, sleeps ``startup_settle``, then polls
        ``subscribe()`` until ``frame_check_count`` consecutive frames show
        strictly advancing ``frame_serial`` for both arms. On success, clears
        errors on both arms in a single transaction.

        Raises
        ------
        ConnectionError
            If ``robot.connect`` fails or the link never produces advancing
            frames within ``frame_check_timeout``.
        """
        with self._lock:
            if self._connected:
                return
            robot = fx_robot.Marvin_Robot()
            dcss = fx_robot.DCSS()
            ok = robot.connect(self.ip)
            if not ok:
                raise ConnectionError(
                    f"Failed to connect to Tianji/Marvin robot at {self.ip}. "
                    "Ensure the controller is reachable (ping) and the network cable is plugged in."
                )
            self._robot = robot
            self._dcss = dcss
            # Mark connected first so subscribe()/transaction() work during the
            # validation + error-clear below; reset on failure in `except`.
            self._connected = True
            try:
                time.sleep(self._startup_settle)
                self._validate_link()
                # Clear errors on both arms in one atomic transaction.
                with self.transaction() as r:
                    r.clear_error("A")
                    r.clear_error("B")
            except Exception:
                self._connected = False
                self._best_effort_release()
                self._robot = None
                self._dcss = None
                raise

    def _validate_link(self) -> None:
        """Poll subscribe() until frame_serial advances for both arms."""
        deadline = time.monotonic() + self._frame_check_timeout
        prev = [None, None]
        streak = 0
        while True:
            data = self._robot.subscribe(self._dcss)
            outputs = data["outputs"]
            serials = [int(outputs[0]["frame_serial"]), int(outputs[1]["frame_serial"])]
            advanced = (
                prev[0] is not None
                and prev[1] is not None
                and serials[0] > prev[0]
                and serials[1] > prev[1]
            )
            if advanced:
                streak += 1
                if streak >= self._frame_check_count:
                    return
            else:
                streak = 0
            prev = serials
            if time.monotonic() >= deadline:
                raise ConnectionError(
                    f"Tianji link at {self.ip} did not produce {self._frame_check_count} "
                    f"consecutive advancing frames within {self._frame_check_timeout}s "
                    f"(last frame_serial={serials})."
                )
            time.sleep(0.005)

    def _best_effort_release(self) -> None:
        """Best-effort release of the robot handle; never raises."""
        robot = self._robot
        if robot is None:
            return
        try:
            robot.release_robot()
        except Exception:
            pass

    def close(self) -> None:
        """Best-effort release of the controller link.

        Never raises. Safe to call multiple times. After closing, the object
        is disconnected and further ``subscribe`` / ``transaction`` calls raise
        ``RuntimeError``.
        """
        with self._lock:
            if not self._connected:
                return
            self._best_effort_release()
            self._connected = False
            self._robot = None
            self._dcss = None

    # ========== Public API ==========
    def subscribe(self) -> Dict:
        """Return the latest subscribed state dict for both arms.

        Raises
        ------
        RuntimeError
            If the connection is not established.
        """
        with self._lock:
            if not self._connected:
                raise RuntimeError("TianjiConnection is not connected.")
            return self._robot.subscribe(self._dcss)

    @contextmanager
    def transaction(self) -> Generator[fx_robot.Marvin_Robot, None, None]:
        """Context manager for an atomic command transaction.

        Acquires the connection lock, calls ``robot.clear_set()``, then yields
        the ``Marvin_Robot`` instance so the caller can issue one or more
        command calls (e.g. ``set_state``, ``set_joint_cmd_pose``). On clean
        exit, ``robot.send_cmd()`` is called. If the block raises, ``send_cmd``
        is skipped so a half-built command is never sent.

        Example
        -------
        >>> with conn.transaction() as r:
        ...     r.set_vel_acc("A", 3, 3)
        ...     r.set_state("A", 1)

        Raises
        ------
        RuntimeError
            If the connection is not established.
        """
        with self._lock:
            if not self._connected:
                raise RuntimeError("TianjiConnection is not connected.")
            self._robot.clear_set()
            try:
                yield self._robot
            except BaseException:
                # Do not send a partially-built command buffer.
                raise
            else:
                self._robot.send_cmd()