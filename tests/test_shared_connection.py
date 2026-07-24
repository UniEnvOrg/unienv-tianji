"""Offline unit tests for the shared TianjiConnection refactor.

These tests never touch real hardware. They use a ``FakeConnection`` stub
that mimics :class:`unienv_tianji.TianjiConnection`'s public surface
(``connected``, ``subscribe()``, ``transaction()``, ``close()``) and records
the per-arm command calls issued inside ``transaction`` blocks.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import pytest

from unienv_tianji import TianjiArmActor, TianjiConnection


class FakeRobot:
    """Records command calls per arm, like the real Marvin_Robot surface."""

    def __init__(self):
        self.calls: List[tuple] = []
        self.clear_set_calls = 0
        self.send_cmd_calls = 0
        self._current_txn: Optional[List[tuple]] = None

    def clear_set(self):
        self.clear_set_calls += 1

    def send_cmd(self):
        self.send_cmd_calls += 1

    def _record(self, call: tuple) -> None:
        self.calls.append(call)
        if self._current_txn is not None:
            self._current_txn.append(call)

    def clear_error(self, arm):
        self._record(("clear_error", arm))

    def set_vel_acc(self, arm, vel, acc):
        self._record(("set_vel_acc", arm, vel, acc))

    def set_state(self, arm, state):
        self._record(("set_state", arm, state))

    def set_joint_cmd_pose(self, arm, joints):
        self._record(("set_joint_cmd_pose", arm, list(joints)))

    def set_impedance_type(self, arm, type):
        self._record(("set_impedance_type", arm, int(type)))

    def set_joint_kd_params(self, arm, K, D):
        self._record(("set_joint_kd_params", arm, list(K), list(D)))

    def set_cart_kd_params(self, arm, K, D, type):
        self._record(("set_cart_kd_params", arm, list(K), list(D), int(type)))

    def set_tool(self, arm, kineParams, dynamicParams):
        self._record(("set_tool", arm, list(kineParams), list(dynamicParams)))


class FakeConnection:
    """Stand-in for TianjiConnection used by the actor."""

    def __init__(self):
        self._robot = FakeRobot()
        self._closed = False
        self._frame_serial = 0
        self.transaction_count = 0
        # Per-transaction call lists, in issue order. transactions[i] is the
        # list of robot calls made inside the i-th transaction block.
        self.transactions: List[List[tuple]] = []

    @property
    def connected(self) -> bool:
        return not self._closed

    def subscribe(self) -> Dict:
        if self._closed:
            raise RuntimeError("FakeConnection is not connected.")
        self._frame_serial += 1
        # Reflect the most recent set_state per arm so _wait_for_state completes.
        states = []
        for arm in ("A", "B"):
            last = None
            for c in self._robot.calls:
                if c[0] == "set_state" and c[1] == arm:
                    last = c[2]
            states.append({"cur_state": last if last is not None else 1, "err_code": 0})
        return {
            "outputs": [
                {
                    "frame_serial": self._frame_serial,
                    "fb_joint_pos": [0.0] * 7,
                    "fb_joint_vel": [0.0] * 7,
                    "fb_joint_sToq": [0.0] * 7,
                },
                {
                    "frame_serial": self._frame_serial,
                    "fb_joint_pos": [0.0] * 7,
                    "fb_joint_vel": [0.0] * 7,
                    "fb_joint_sToq": [0.0] * 7,
                },
            ],
            "states": states,
        }

    @contextmanager
    def transaction(self):
        if self._closed:
            raise RuntimeError("FakeConnection is not connected.")
        self.transaction_count += 1
        txn: List[tuple] = []
        self.transactions.append(txn)
        self._robot._current_txn = txn
        self._robot.clear_set()
        yield self._robot
        self._robot._current_txn = None
        self._robot.send_cmd()

    def close(self) -> None:
        self._closed = True

    # ----- helpers for assertions -----
    def calls_for(self, arm: str) -> List[tuple]:
        return [c for c in self._robot.calls if len(c) >= 2 and c[1] == arm]

    def txn_index_of(self, predicate, arm: Optional[str] = None) -> Optional[int]:
        """Return the index of the first transaction containing a call matching
        ``predicate(call)`` (optionally restricted to ``arm``). None if absent."""
        for i, txn in enumerate(self.transactions):
            for c in txn:
                if arm is not None and (len(c) < 2 or c[1] != arm):
                    continue
                if predicate(c):
                    return i
        return None


def test_connect_true_without_connection_raises():
    with pytest.raises(ValueError, match="requires a shared TianjiConnection"):
        TianjiArmActor(connect=True, connection=None)


def test_two_arms_share_connection_enable():
    conn = FakeConnection()
    actor_a = TianjiArmActor(arm="A", connect=True, connection=conn, post_enable_settle=0.0)
    actor_b = TianjiArmActor(arm="B", connect=True, connection=conn, post_enable_settle=0.0)
    try:
        # Each arm should have issued clear_error, set_vel_acc, set_state for itself.
        a_calls = conn.calls_for("A")
        b_calls = conn.calls_for("B")
        assert any(c[0] == "clear_error" for c in a_calls)
        assert any(c[0] == "set_vel_acc" and c[2] == 6 and c[3] == 6 for c in a_calls)
        assert any(c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_POSITION for c in a_calls)
        assert any(c[0] == "clear_error" for c in b_calls)
        assert any(c[0] == "set_vel_acc" for c in b_calls)
        assert any(c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_POSITION for c in b_calls)
        # No command for the other arm leaked into this arm's enable.
        a_state_calls = [c for c in a_calls if c[0] == "set_state"]
        assert all(c[2] == TianjiArmActor.ARM_STATE_POSITION for c in a_state_calls)
    finally:
        actor_a.close()
        actor_b.close()
        conn.close()


def test_send_joint_command_single_transaction_for_arm_a():
    conn = FakeConnection()
    actor_a = TianjiArmActor(arm="A", connect=True, connection=conn, post_enable_settle=0.0)
    actor_b = TianjiArmActor(arm="B", connect=True, connection=conn, post_enable_settle=0.0)
    try:
        before = conn.transaction_count
        actor_a.send_joint_command(np.zeros(7, dtype=np.float32))
        after = conn.transaction_count
        # Exactly one new transaction for the joint command.
        assert after - before == 1
        # The single transaction issued set_joint_cmd_pose for A only.
        pose_calls_a = [c for c in conn.calls_for("A") if c[0] == "set_joint_cmd_pose"]
        pose_calls_b = [c for c in conn.calls_for("B") if c[0] == "set_joint_cmd_pose"]
        assert len(pose_calls_a) == 1
        assert len(pose_calls_b) == 0
    finally:
        actor_a.close()
        actor_b.close()
        conn.close()


def test_closing_actor_a_does_not_invalidate_connection_for_b():
    conn = FakeConnection()
    actor_a = TianjiArmActor(arm="A", connect=True, connection=conn, post_enable_settle=0.0)
    actor_b = TianjiArmActor(arm="B", connect=True, connection=conn, post_enable_settle=0.0)
    try:
        # Close A: should set A to IDLE via a transaction, but NOT close the connection.
        actor_a.close()
        assert conn.connected, "Closing actor A must not close the shared connection."
        idle_calls_a = [c for c in conn.calls_for("A") if c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_IDLE]
        assert len(idle_calls_a) >= 1
        # B can still read observations through the shared connection.
        actor_b.post_environment_step(0.04)
        obs = actor_b.get_observation()
        assert obs is not None
        assert obs["joint_positions"].shape == (7,)
    finally:
        actor_a.close()
        actor_b.close()
        conn.close()


def test_tianji_connection_offline_construct_and_behaviour():
    conn = TianjiConnection("192.168.1.190", connect=False)
    try:
        assert conn.connected is False
        with pytest.raises(RuntimeError):
            conn.subscribe()
        with pytest.raises(RuntimeError):
            with conn.transaction():
                pass
        # close() must be safe even when never connected.
        conn.close()
        conn.close()  # idempotent
    finally:
        conn.close()


# ========== Control-mode tests ==========

def test_invalid_control_mode_raises():
    with pytest.raises(ValueError, match="control_mode must be one of"):
        TianjiArmActor(connect=False, control_mode="bogus")


def test_joint_impedance_enable_order_and_params():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode="joint_impedance", post_enable_settle=0.0,
    )
    try:
        a_calls = conn.calls_for("A")
        # clear_error, set_state(3), set_impedance_type(1), set_joint_kd_params, set_tool all called.
        assert any(c[0] == "clear_error" for c in a_calls)
        assert any(c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_TORQ for c in a_calls)
        assert any(c[0] == "set_impedance_type" and c[2] == 1 for c in a_calls)
        assert any(c[0] == "set_joint_kd_params" for c in a_calls)
        assert any(c[0] == "set_tool" for c in a_calls)

        # Transaction ORDER: clear_error appears in an earlier transaction than set_state(3).
        clear_idx = conn.txn_index_of(lambda c: c[0] == "clear_error", arm="A")
        state_idx = conn.txn_index_of(lambda c: c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_TORQ, arm="A")
        assert clear_idx is not None and state_idx is not None
        assert clear_idx < state_idx

        # set_tool dynamic params: [mass=1.40, mx=0, my=0, mz=80, Ixx..Izz=0]*6
        tool_calls = [c for c in a_calls if c[0] == "set_tool"]
        assert len(tool_calls) == 1
        dyn = tool_calls[0][3]
        assert dyn == [1.40, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    finally:
        actor.close()
        conn.close()


def test_cart_impedance_enable_kd_before_state():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode="cart_impedance", post_enable_settle=0.0,
    )
    try:
        a_calls = conn.calls_for("A")
        # set_cart_kd_params called with type=2.
        cart_calls = [c for c in a_calls if c[0] == "set_cart_kd_params"]
        assert len(cart_calls) == 1
        assert cart_calls[0][4] == 2  # type
        # set_impedance_type(2) called.
        assert any(c[0] == "set_impedance_type" and c[2] == 2 for c in a_calls)
        # set_state(3) called.
        assert any(c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_TORQ for c in a_calls)

        # set_cart_kd_params appears in an earlier transaction than set_state(3).
        cart_idx = conn.txn_index_of(lambda c: c[0] == "set_cart_kd_params", arm="A")
        state_idx = conn.txn_index_of(lambda c: c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_TORQ, arm="A")
        assert cart_idx is not None and state_idx is not None
        assert cart_idx < state_idx
    finally:
        actor.close()
        conn.close()


def test_position_mode_no_impedance_or_tool_calls():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode="position", post_enable_settle=0.0,
    )
    try:
        a_calls = conn.calls_for("A")
        assert not any(c[0] == "set_impedance_type" for c in a_calls)
        assert not any(c[0] == "set_joint_kd_params" for c in a_calls)
        assert not any(c[0] == "set_cart_kd_params" for c in a_calls)
        assert not any(c[0] == "set_tool" for c in a_calls)
        # Still enters position state.
        assert any(c[0] == "set_state" and c[2] == TianjiArmActor.ARM_STATE_POSITION for c in a_calls)
    finally:
        actor.close()
        conn.close()


class FakeKine:
    """Stand-in for Marvin_Kine's ik() surface used by send_eef_command."""

    def __init__(self, ik_result):
        # ik_result: either False, or a fake structure-like object exposing
        # m_Output_RetJoint.to_list() -> list[7] of degrees.
        self._ik_result = ik_result
        self.ik_calls = 0

    def ik(self, structure_data):
        self.ik_calls += 1
        return self._ik_result


class _FakeRetJoint:
    def to_list(self):
        return [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]


class _FakeIkResult:
    m_Output_RetJoint = _FakeRetJoint()


def test_send_eef_command_uses_ik_then_send_joint_command():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode="position", post_enable_settle=0.0,
    )
    # Inject a fake kinematics helper (bypasses _init_kine / native lib).
    actor._kine = FakeKine(ik_result=_FakeIkResult())
    try:
        before = conn.transaction_count
        actor.send_eef_command(np.eye(4, dtype=np.float64))
        after = conn.transaction_count
        # Exactly one transaction: the set_joint_cmd_pose from send_joint_command.
        assert after - before == 1
        # The fake IK was called once.
        assert actor._kine.ik_calls == 1
        # set_joint_cmd_pose for A received the IK degrees (converted to rad then
        # back to deg by send_joint_command, so the values should round-trip).
        pose_calls = [c for c in conn.calls_for("A") if c[0] == "set_joint_cmd_pose"]
        assert len(pose_calls) == 1
        sent_deg = pose_calls[0][2]
        assert len(sent_deg) == 7
        np.testing.assert_allclose(sent_deg, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], atol=1e-3)
    finally:
        actor.close()
        conn.close()


def test_send_eef_command_ik_failure_raises_and_sends_nothing():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode="position", post_enable_settle=0.0,
    )
    actor._kine = FakeKine(ik_result=False)
    try:
        before = conn.transaction_count
        with pytest.raises(RuntimeError, match="IK failed"):
            actor.send_eef_command(np.eye(4, dtype=np.float64))
        after = conn.transaction_count
        # No command transaction was issued.
        assert after == before
        assert actor._kine.ik_calls == 1
    finally:
        actor.close()
        conn.close()


def test_send_eef_command_without_kine_raises():
    actor = TianjiArmActor(connect=False, control_mode="position")
    try:
        with pytest.raises(RuntimeError, match="requires a kinematics helper"):
            actor.send_eef_command(np.eye(4, dtype=np.float64))
    finally:
        actor.close()