"""Process output queued while a Desktop turn is busy must not wake it after exit."""
import threading

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import format_process_notification
from tui_gateway import server


@pytest.fixture
def delivery(monkeypatch):
    registry = ProcessRegistry()
    process = ProcessSession(id="proc_heartbeat", command="owned bounded task")
    registry._running[process.id] = process
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    emitted, submitted, frames = set(), [], []
    monkeypatch.setattr(server, "_emit", lambda *args: frames.append(args))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kwargs: submitted.append(args) or True)
    monkeypatch.setattr(server, "_background_notifications_off", lambda session: False)
    return registry, process, session, emitted, submitted, frames


def _deliver(delivery, event, deferred=None):
    registry, _process, session, emitted, _submitted, _frames = delivery
    return server._notif_handle_event("owner", session, event, emitted, registry,
                                      format_process_notification, deferred, owned=True)


def _event(kind="heartbeat", exit_code=0):
    return {"type": kind, "session_id": "proc_heartbeat", "seq": 1,
            "command": "owned bounded task", "output": "new output", "exit_code": exit_code}


@pytest.mark.parametrize("kind,exited,exit_code,expected", [
    ("heartbeat", False, None, 1),
    ("heartbeat", True, 0, 0),
    ("heartbeat", True, 1, 0),
    ("completion", True, 0, 1),
    ("completion", True, 1, 1),
])
def test_lifecycle_filters_heartbeats_without_losing_terminal_results(delivery, kind, exited, exit_code, expected):
    _registry, process, _session, _emitted, submitted, frames = delivery
    process.exited = exited
    process.exit_code = exit_code
    assert _deliver(delivery, _event(kind, exit_code)) is True
    assert len(submitted) == expected
    if kind == "heartbeat" and exited:
        assert frames == []
    if kind == "completion" and exit_code == 1:
        assert "exit code 1" in submitted[0][3]


def test_busy_heartbeat_is_dropped_after_process_finishes(delivery):
    registry, process, session, _emitted, submitted, _frames = delivery
    session["running"] = True
    assert _deliver(delivery, _event(), deferred=[]) is False
    process.exited = True
    session["running"] = False
    assert _deliver(delivery, registry.completion_queue.get_nowait()) is True
    assert submitted == []
    assert session["running"] is False
    assert registry.completion_queue.empty()


def test_redelivery_does_not_start_a_second_turn_but_new_output_does(delivery):
    _registry, _process, session, _emitted, submitted, _frames = delivery
    event = _event()
    assert _deliver(delivery, event) is True
    session["running"] = False
    assert _deliver(delivery, dict(event)) is True
    assert len(submitted) == 1
    assert _deliver(delivery, {**event, "seq": 2, "output": "later output"}) is True
    assert len(submitted) == 2


def test_exit_between_queue_read_and_turn_claim_releases_the_turn(delivery, monkeypatch):
    _registry, process, session, _emitted, submitted, _frames = delivery
    claim_turn = server._notif_claim_turn
    def claim_then_exit(owner):
        claimed = claim_turn(owner)
        process.exited = True
        return claimed
    monkeypatch.setattr(server, "_notif_claim_turn", claim_then_exit)
    assert _deliver(delivery, _event()) is True
    assert submitted == []
    assert session["running"] is False
