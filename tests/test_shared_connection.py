"""Offline unit tests for the shared TianjiConnection refactor.

These tests never touch real hardware. They use a ``FakeConnection`` stub
that mimics :class:`unienv_tianji.TianjiConnection`'s public surface
(``connected``, ``subscribe()``, ``transaction()``, ``close()``) and records
the per-arm command calls issued inside ``transaction`` blocks.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from unienv_tianji import TianjiArmActor, TianjiConnection, TianjiArmHardwareError


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


def _fake_arm_outputs(frame_serial, fb_joint_pos=None):
    """A full per-arm RT_OUT feedback dict matching fx_robot.subscribe's output."""
    return {
        "frame_serial": frame_serial,
        "fb_joint_pos": fb_joint_pos if fb_joint_pos is not None else [10.0] + [0.0] * 6,
        "fb_joint_vel": [0.0] * 7,
        "fb_joint_posE": [0.0] * 7,
        "fb_joint_cmd": [0.0] * 7,
        "fb_joint_cToq": [0.0] * 7,
        "fb_joint_sToq": [0.0] * 7,
        "fb_joint_them": [0.0] * 7,
        "est_joint_firc": [0.0] * 7,
        "est_joint_firc_dot": [0.0] * 7,
        "est_joint_force": [0.0] * 7,
        "est_cart_fn": [0.0] * 6,
    }


class FakeConnection:
    """Stand-in for TianjiConnection used by the actor."""

    def __init__(
        self,
        *,
        staged_application: bool = False,
        mismatch: Optional[str] = None,
        feedback_error_code: int = 0,
        feedback_positions: Optional[List[List[float]]] = None,
    ):
        self._robot = FakeRobot()
        self._closed = False
        self._frame_serial = 0
        self.subscribe_count = 0
        self._staged_application = staged_application
        self._mismatch = mismatch
        self._feedback_error_code = feedback_error_code
        self._feedback_positions = feedback_positions
        self._post_config_subscribe_count = 0
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
        self.subscribe_count += 1
        configured = any(c[0] == "set_impedance_type" for c in self._robot.calls)
        if configured:
            self._post_config_subscribe_count += 1
        self._frame_serial += 1
        if self._feedback_positions is None:
            fb_joint_pos = [10.0] + [0.0] * 6
        else:
            fb_joint_pos = self._feedback_positions[
                min(self.subscribe_count - 1, len(self._feedback_positions) - 1)
            ]
        # Reflect the most recent set_state per arm so _wait_for_state completes.
        states = []
        inputs = []
        for arm in ("A", "B"):
            last = None
            imp_type = 0
            joint_k = [0.0] * 7
            joint_d = [0.0] * 7
            cart_k = [0.0] * 7
            cart_d = [0.0] * 7
            for c in self._robot.calls:
                if c[0] == "set_state" and c[1] == arm:
                    last = c[2]
                elif c[0] == "set_impedance_type" and c[1] == arm:
                    imp_type = c[2]
                elif c[0] == "set_joint_kd_params" and c[1] == arm:
                    joint_k, joint_d = c[2], c[3]
                elif c[0] == "set_cart_kd_params" and c[1] == arm:
                    cart_k, cart_d = c[2], c[3]
            if self._staged_application and self._post_config_subscribe_count == 1:
                imp_type = 0
                joint_k = [0.0] * 7
                joint_d = [0.0] * 7
                cart_k = [0.0] * 7
                cart_d = [0.0] * 7
            elif self._staged_application and self._post_config_subscribe_count == 2:
                joint_k = [0.0] * 7
                joint_d = [0.0] * 7
                cart_k = [0.0] * 7
                cart_d = [0.0] * 7
            states.append({"cur_state": last if last is not None else 1, "err_code": 0})
            inputs.append({
                "imp_type": imp_type,
                "joint_k": joint_k,
                "joint_d": joint_d,
                "cart_k": cart_k[:6],
                "cart_d": cart_d[:6],
                "cart_kn": cart_k[6],
                "cart_dn": cart_d[6],
            })
        data = {
            "outputs": [
                _fake_arm_outputs(self._frame_serial, fb_joint_pos),
                _fake_arm_outputs(self._frame_serial, fb_joint_pos),
            ],
            "states": states,
            "inputs": inputs,
        }
        if (
            self._mismatch is not None
            and configured
            # Let _wait_for_state see one healthy post-config frame first.
            and self._post_config_subscribe_count > 1
        ):
            if self._mismatch == "state":
                data["states"][0]["cur_state"] = TianjiArmActor.ARM_STATE_POSITION
            elif self._mismatch == "type":
                data["inputs"][0]["imp_type"] = (
                    2 if data["inputs"][0]["imp_type"] == 1 else 1
                )
            elif self._mismatch == "gains":
                if data["inputs"][0]["imp_type"] == 1:
                    data["inputs"][0]["joint_k"][0] = 0.0
                else:
                    data["inputs"][0]["cart_k"][0] = 0.0
        if configured and self._feedback_error_code and self._post_config_subscribe_count > 1:
            data["states"][0]["err_code"] = self._feedback_error_code
        return data

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
        # State-wait read plus impedance verification read.
        assert conn.subscribe_count >= 2

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


@pytest.mark.parametrize("control_mode", ["joint_impedance", "cart_impedance"])
def test_impedance_waits_for_staged_controller_values(control_mode):
    # The controller first reports state, then impedance type, then gains. The
    # verifier must keep polling until the complete configuration is visible.
    conn = FakeConnection(staged_application=True)
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode=control_mode, post_enable_settle=0.0,
    )
    try:
        assert conn._post_config_subscribe_count >= 3
    finally:
        actor.close()
        conn.close()


def test_joint_impedance_partial_feedback_timeout_reports_last_values():
    conn = FakeConnection(mismatch="gains")
    with pytest.raises(TianjiArmHardwareError, match="impedance verification failed") as excinfo:
        TianjiArmActor(
            arm="A", connect=True, connection=conn,
            control_mode="joint_impedance", post_enable_settle=0.0,
        )
    assert "last observed" in str(excinfo.value)
    conn.close()


def test_constructor_enable_failure_returns_arm_to_idle():
    conn = FakeConnection(mismatch="gains")
    with pytest.raises(TianjiArmHardwareError, match="impedance verification failed"):
        TianjiArmActor(
            arm="A", connect=True, connection=conn,
            control_mode="joint_impedance", post_enable_settle=0.0,
        )
    assert any(
        c == ("set_state", "A", TianjiArmActor.ARM_STATE_IDLE)
        for c in conn.calls_for("A")
    )
    conn.close()


def test_reenable_failure_returns_arm_to_idle():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode="joint_impedance", post_enable_settle=0.0,
    )
    try:
        idle_calls_before = len([
            c for c in conn.calls_for("A")
            if c == ("set_state", "A", TianjiArmActor.ARM_STATE_IDLE)
        ])
        conn._mismatch = "gains"
        with pytest.raises(TianjiArmHardwareError, match="impedance verification failed"):
            actor.set_enabled(True)
        idle_calls_after = len([
            c for c in conn.calls_for("A")
            if c == ("set_state", "A", TianjiArmActor.ARM_STATE_IDLE)
        ])
        assert idle_calls_after == idle_calls_before + 1
    finally:
        actor.close()
        conn.close()


@pytest.mark.parametrize("control_mode", ["joint_impedance", "cart_impedance"])
@pytest.mark.parametrize("mismatch", ["state", "type", "gains"])
def test_impedance_feedback_mismatch_raises_at_deadline(control_mode, mismatch):
    conn = FakeConnection(mismatch=mismatch)
    with pytest.raises(TianjiArmHardwareError, match="impedance verification failed") as excinfo:
        TianjiArmActor(
            arm="A", connect=True, connection=conn,
            control_mode=control_mode, post_enable_settle=0.0,
        )
    # The state-wait frame is healthy; later partial values persist. Reaching
    # several polls proves mismatch is held to the verification deadline.
    assert conn._post_config_subscribe_count > 3
    assert "last observed" in str(excinfo.value)
    conn.close()


@pytest.mark.parametrize("control_mode", ["joint_impedance", "cart_impedance"])
def test_impedance_feedback_error_blocks_success(control_mode):
    conn = FakeConnection(feedback_error_code=42)
    with pytest.raises(TianjiArmHardwareError, match="err_code=42") as excinfo:
        TianjiArmActor(
            arm="A", connect=True, connection=conn,
            control_mode=control_mode, post_enable_settle=0.0,
        )
    assert excinfo.value.error_code == 42
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
        assert conn.subscribe_count == 6  # state wait + five valid feedback frames
    finally:
        actor.close()
        conn.close()


def test_enable_waits_for_consecutive_valid_joint_feedback():
    zero = [0.0] * 7
    valid = [10.0] + [0.0] * 6
    # The first zero frame is consumed by _wait_for_state; the gate then sees
    # two zero frames before its required two consecutive valid frames.
    conn = FakeConnection(feedback_positions=[zero, zero, zero, valid, valid])
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
        feedback_valid_frames=2,
    )
    try:
        assert conn.subscribe_count == 5
    finally:
        actor.close()
        conn.close()


def test_valid_feedback_counter_resets_after_invalid_frame():
    valid = [10.0] + [0.0] * 6
    invalid = [0.0] * 7
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
        feedback_valid_frames=3,
    )
    try:
        conn._feedback_positions = [valid, valid, invalid, valid, valid, valid]
        conn.subscribe_count = 0
        actor._wait_for_valid_feedback()
        # The first two valid frames cannot count across the invalid frame.
        assert conn.subscribe_count == 6
    finally:
        actor.close()
        conn.close()


def test_valid_feedback_counter_reset_times_out_without_enough_new_frames():
    valid = [10.0] + [0.0] * 6
    invalid = [0.0] * 7
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
        feedback_valid_frames=3,
    )
    try:
        conn._feedback_positions = [valid, valid, invalid, valid, valid]
        conn.subscribe_count = 0
        actor.feedback_valid_timeout = 0.1
        with pytest.raises(TianjiArmHardwareError, match="timed out waiting for valid joint feedback"):
            actor._wait_for_valid_feedback()
        assert conn.subscribe_count == 5
    finally:
        actor.close()
        conn.close()


def test_enable_rejects_persistently_zero_joint_feedback_and_idles():
    conn = FakeConnection(feedback_positions=[[0.0] * 7])
    with pytest.raises(TianjiArmHardwareError, match="timed out waiting for valid joint feedback") as excinfo:
        TianjiArmActor(
            arm="A", connect=True, connection=conn, post_enable_settle=0.0,
            feedback_valid_frames=1, feedback_valid_timeout=0.0,
        )
    assert "fb_joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" in str(excinfo.value)
    assert any(
        c == ("set_state", "A", TianjiArmActor.ARM_STATE_IDLE)
        for c in conn.calls_for("A")
    )
    conn.close()


def test_enable_valid_feedback_from_first_gate_frame_has_no_poll_delay():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
        feedback_valid_frames=1,
    )
    try:
        # One state-wait poll plus one immediately-successful validity poll.
        assert conn.subscribe_count == 2
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


def test_send_eef_command_offline_without_kine_is_noop():
    # With the offline no-op reorder, send_eef_command(connect=False) returns
    # early (consistent with send_joint_command being a no-op offline) instead
    # of raising "requires a kinematics helper".
    actor = TianjiArmActor(connect=False, control_mode="position")
    try:
        # No raise, no-op.
        actor.send_eef_command(np.eye(4, dtype=np.float64))
    finally:
        actor.close()


def test_send_eef_command_connected_without_kine_raises():
    conn = FakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn,
        control_mode="position", post_enable_settle=0.0,
    )
    try:
        with pytest.raises(RuntimeError, match="requires a kinematics helper"):
            actor.send_eef_command(np.eye(4, dtype=np.float64))
    finally:
        actor.close()
        conn.close()


# ========== Hardware-error raising (Part B) ==========

class ErrorFakeConnection(FakeConnection):
    """FakeConnection whose subscribe() can inject per-arm error state on demand.

    Starts healthy (so the actor's _enable_arm / _wait_for_state completes),
    then ``inject_error`` flips the per-arm state to the injected values for
    subsequent subscribe() calls.
    """

    def __init__(self):
        super().__init__()
        self._injected: Optional[Dict[str, Tuple[int, int]]] = None

    def inject_error(self, arm_states: Dict[str, Tuple[int, int]]) -> None:
        self._injected = arm_states

    def subscribe(self) -> Dict:
        if self._closed:
            raise RuntimeError("FakeConnection is not connected.")
        self._frame_serial += 1
        states = []
        for arm in ("A", "B"):
            if self._injected is not None and arm in self._injected:
                cur, err = self._injected[arm]
            else:
                # Healthy: reflect the most recent set_state per arm.
                last = None
                for c in self._robot.calls:
                    if c[0] == "set_state" and c[1] == arm:
                        last = c[2]
                cur, err = (last if last is not None else 1), 0
            states.append({"cur_state": cur, "err_code": err})
        return {
            "outputs": [
                _fake_arm_outputs(self._frame_serial),
                _fake_arm_outputs(self._frame_serial),
            ],
            "states": states,
        }


def test_error_code_nonzero_raises_on_feedback_refresh():
    conn = ErrorFakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
    )
    try:
        # Enable completed healthy; now inject an error code for arm A.
        conn.inject_error({"A": (TianjiArmActor.ARM_STATE_POSITION, 42)})
        with pytest.raises(TianjiArmHardwareError) as excinfo:
            actor.post_environment_step(0.04)
        assert excinfo.value.arm == "A"
        assert excinfo.value.error_code == 42
    finally:
        actor.close()
        conn.close()


def test_error_state_100_raises_on_feedback_refresh():
    conn = ErrorFakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
    )
    try:
        conn.inject_error({"A": (TianjiArmActor.ARM_STATE_ERROR, 0)})
        with pytest.raises(TianjiArmHardwareError) as excinfo:
            actor.post_environment_step(0.04)
        assert excinfo.value.cur_state == TianjiArmActor.ARM_STATE_ERROR
    finally:
        actor.close()
        conn.close()


def test_normal_state_no_error_does_not_raise():
    conn = ErrorFakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
    )
    try:
        # Healthy (no injection): post-step must not raise.
        actor.post_environment_step(0.04)
        obs = actor.get_observation()
        assert obs is not None
        assert "joint_positions" in obs
        assert "arm_state" not in obs
        assert "error_code" not in obs
    finally:
        actor.close()
        conn.close()


# ========== Extended proprioception channels ==========

EXTENDED_KEYS = {
    "joint_currents",
    "joint_temperatures",
    "joint_friction_estimates",
    "joint_external_force_estimates",
    "cartesian_force_estimate",
}


def test_extended_proprioception_keys_present_by_default():
    actor = TianjiArmActor(connect=False)
    try:
        keys = set(actor.observation_space.spaces.keys())
        # Basic trio always present.
        assert {"joint_positions", "joint_velocities", "joint_torques"} <= keys
        # Extended channels on by default.
        assert EXTENDED_KEYS <= keys
        # Shapes.
        assert actor.observation_space.spaces["joint_currents"].shape == (7,)
        assert actor.observation_space.spaces["joint_temperatures"].shape == (7,)
        assert actor.observation_space.spaces["joint_friction_estimates"].shape == (7,)
        assert actor.observation_space.spaces["joint_external_force_estimates"].shape == (7,)
        assert actor.observation_space.spaces["cartesian_force_estimate"].shape == (6,)
    finally:
        actor.close()


def test_extended_proprioception_offline_zeros():
    actor = TianjiArmActor(connect=False)
    try:
        actor.after_reset()
        obs = actor.get_observation()
        for k in EXTENDED_KEYS:
            assert k in obs, f"missing {k}"
            assert obs[k] is not None
            np.testing.assert_array_equal(obs[k], 0.0)
        # cartesian_force_estimate is (6,), the rest (7,).
        assert obs["cartesian_force_estimate"].shape == (6,)
        for k in EXTENDED_KEYS - {"cartesian_force_estimate"}:
            assert obs[k].shape == (7,)
    finally:
        actor.close()


def test_extended_proprioception_flows_through_connected_fake():
    conn = ErrorFakeConnection()
    actor = TianjiArmActor(
        arm="A", connect=True, connection=conn, post_enable_settle=0.0,
    )
    try:
        actor.post_environment_step(0.04)
        obs = actor.get_observation()
        for k in EXTENDED_KEYS:
            assert k in obs
            assert obs[k] is not None
    finally:
        actor.close()
        conn.close()


def test_extended_proprioception_disabled_hides_keys():
    actor = TianjiArmActor(connect=False, read_extended_proprioception=False)
    try:
        keys = set(actor.observation_space.spaces.keys())
        assert {"joint_positions", "joint_velocities", "joint_torques"} <= keys
        for k in EXTENDED_KEYS:
            assert k not in keys
        actor.after_reset()
        obs = actor.get_observation()
        for k in EXTENDED_KEYS:
            assert k not in obs
    finally:
        actor.close()


def test_offline_never_raises():
    actor = TianjiArmActor(connect=False)
    try:
        actor.after_reset()
        obs = actor.get_observation()
        assert obs is not None
        actor.post_environment_step(0.04)
        obs2 = actor.get_observation()
        assert obs2 is not None
    finally:
        actor.close()
