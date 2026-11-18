"""A notification that claims the session's turn must hand it back whenever no turn runs.

The poller claims the idle session (``running = True``) and only then claims the event's durable
delivery row. Nothing else clears ``running``: a busy session is exempt from the reaper, keeps its
active-session lease, diverts every ``prompt.submit`` into a queue that only a finishing turn
drains, and never polls its bot mailbox again — so a dispatch that returns or raises without
starting a turn leaves that session unusable for the life of the backend.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from tui_gateway import server

DELEGATION = {"type": "async_delegation", "delegation_id": "deleg-1", "session_key": "stored"}


def _claimed_session() -> dict:
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    assert server._notif_claim_turn(session) is True
    return session


def _no_turn(monkeypatch) -> list:
    started: list = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: started.append(a))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    return started


@pytest.mark.parametrize("claim", [None, sqlite3.OperationalError("database is locked")],
                         ids=["row-held-by-another-consumer", "ledger-unreadable"])
def test_a_lost_delivery_claim_hands_the_turn_back(monkeypatch, claim):
    """A gateway sharing this home claims the durable row before it verifies the target, so the
    poller holding the live copy of the same event gets ``None`` — after it already took the turn."""
    def _claim(evt, consumer):
        if isinstance(claim, Exception):
            raise claim
        return claim

    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", _claim)
    started = _no_turn(monkeypatch)
    session = _claimed_session()

    server._notif_dispatch_event("sid", session, dict(DELEGATION), "text")

    assert session["running"] is False
    assert started == []
    assert server._notif_claim_turn(session) is True, "the session must be claimable again"


@pytest.mark.parametrize("fail_at", ["claim", "render"])
def test_a_completion_batch_that_cannot_be_prepared_hands_the_turn_back(monkeypatch, fail_at):
    events = [{"type": "completion", "session_id": "proc_a"}, {"type": "completion", "session_id": "proc_b"}]
    released: list = []
    claims = iter(["claim-a", sqlite3.OperationalError("database is locked")] if fail_at == "claim"
                  else ["claim-a", "claim-b"])

    def _claim(evt, consumer):
        value = next(claims)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", _claim)
    monkeypatch.setattr("tools.async_delegation.release_event_delivery", lambda evt, c: released.append(c))
    if fail_at == "render":
        monkeypatch.setattr("tools.process_registry_notifications.ProcessNotificationBatch.render",
                            lambda self, registry: (_ for _ in ()).throw(ValueError("bad payload")))
    started = _no_turn(monkeypatch)
    session = {"history_lock": threading.RLock(), "running": False, "history": []}

    server._notif_dispatch_completions("sid", session, [(e, "t") for e in events],
                                       SimpleNamespace(completion_queue=queue.Queue()), None)

    assert session["running"] is False and started == []
    # Claims already taken go back too, or those completions are lost to every consumer for 300 s.
    assert released == (["claim-a"] if fail_at == "claim" else ["claim-a", "claim-b"])


def test_a_loop_wakeup_whose_send_cannot_start_hands_the_turn_back(monkeypatch):
    """The /loop slash wakeup re-claims the turn for a command that resolves to a prompt, then runs
    the send under ``except Exception: pass`` — which swallowed the only signal that no turn started."""
    monkeypatch.setitem(server._methods, "command.dispatch",
                        lambda rid, params: {"result": {"type": "send", "message": "run the skill"}})
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no free worker")))
    ticks: list = []
    mgr = SimpleNamespace(abandon_tick=lambda: ticks.append("abandoned"),
                          complete_tick=lambda text: ticks.append("completed") or {})
    session = _claimed_session()

    server._notif_slash_loop_tick("rid", "sid", session, mgr, "/skill go")

    assert session["running"] is False
    assert ticks == ["completed"]


def test_the_poller_thread_survives_a_dispatch_that_raises(monkeypatch):
    """The poller is the session's only path to notifications, /loop, /heartbeat and its bot
    mailbox; an exception out of one event's dispatch used to end the thread for good."""
    events: queue.Queue = queue.Queue()
    events.put({"type": "completion", "session_id": "proc_a"})
    monkeypatch.setattr("tools.process_registry.process_registry",
                        SimpleNamespace(completion_queue=events, restore_completions=lambda: 0))
    for name in ("_poll_bot_live_delivery_guarded", "_maybe_fire_tui_loop_tick",
                 "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    stop = threading.Event()
    handled: list = []

    def _handle_ready(sid, session, ready, emitted, registry, fmt, deferred, **kwargs):
        if deferred is not None:  # the post-stop drain
            return
        handled.append(len(ready))
        if len(handled) == 1:
            events.put({"type": "completion", "session_id": "proc_b"})
            raise RuntimeError("one bad event")
        stop.set()

    monkeypatch.setattr(server, "_notif_handle_ready", _handle_ready)
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    worker = threading.Thread(target=server._notification_poller_scoped_loop, args=(stop, "sid", session), daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert handled == [1, 1], "the second event must still be dispatched after the first one raised"


@pytest.mark.parametrize("shutdown", [False, True])
def test_malformed_event_does_not_drop_valid_completions_in_same_snapshot(monkeypatch, shutdown):
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_notifications import format_process_notification

    registry = ProcessRegistry()
    submitted = _no_turn(monkeypatch)
    monkeypatch.setattr(server, "_background_notifications_off", lambda session: False)
    faults = []
    monkeypatch.setattr(server, "_notif_log_failure", lambda stage, error: faults.append(error))
    session = {"history_lock": threading.RLock(), "running": False, "history": []}
    events = [
        {"type": "completion", "session_id": "proc_ok", "exit_code": 0},
        {"type": "completion", "session_id": "proc_invalid", "exit_code": []},
        {"type": "completion", "session_id": "proc_failed", "exit_code": 1},
    ]
    deferred = [] if shutdown else None

    server._notif_handle_ready("sid", session, events, set(), registry,
                               format_process_notification, deferred, owned=True)

    assert len(faults) == 1 and isinstance(faults[0], TypeError)
    assert len(submitted) == 1
    prompt = submitted[0][3]
    assert "proc_ok" in prompt and "exit code 0" in prompt
    assert "proc_failed" in prompt and "exit code 1" in prompt
    assert "proc_invalid" not in prompt


def test_refused_prompt_keeps_durable_delegation_pending_until_accepted(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import async_delegation as ledger

    token = set_hermes_home_override(str(tmp_path))
    try:
        event = dict(DELEGATION, status="completed")
        ledger._persist_dispatch(dict(event, dispatched_at=1.0))
        ledger._persist_completion(event, {"status": "completed"})
        submitted = []
        monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

        def submit(rid, sid, session, text, **kwargs):
            submitted.append(text)
            if len(submitted) == 1:
                session["running"] = False
                return False
            return True

        monkeypatch.setattr(server, "_run_prompt_submit", submit)
        session = _claimed_session()

        assert server._notif_dispatch_event("sid", session, event, "result") is False
        row = ledger.get_durable_delegation(event["delegation_id"])
        assert row["delivery_state"] == "pending"
        assert row["result"] == {"status": "completed"}
        pending = queue.Queue()
        now = time.time()
        assert ledger.sweep_orphaned_completions(pending, now=now) == 0
        retry_at = now + ledger._ORPHAN_STALE_S + 1
        assert ledger.sweep_orphaned_completions(pending, now=retry_at) == 1
        assert ledger.sweep_orphaned_completions(pending, now=retry_at + 1) == 0
        replay = pending.get_nowait()
        assert replay["delegation_id"] == event["delegation_id"]
        assert server._notif_claim_turn(session) is True

        assert server._notif_dispatch_event("sid", session, replay, "result") is True
        assert ledger.get_durable_delegation(event["delegation_id"])["delivery_state"] == "delivered"
        assert ledger.claim_event_delivery(event, "duplicate-check") is None
        assert submitted == ["result", "result"]
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("accepted", [False, True])
def test_notification_start_frame_follows_prompt_admission_once(monkeypatch, accepted):
    import contextlib

    frames = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, *args: frames.append((event, sid)))
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: True)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda session: contextlib.nullcontext())
    monkeypatch.setattr(server, "_routing_provenance_db", lambda session: contextlib.nullcontext())
    monkeypatch.setattr(server, "_admit_prompt_turn",
                        lambda *args: ([], SimpleNamespace(session_id="test")) if accepted else None)
    monkeypatch.setattr(server, "_start_session_work", lambda *args, **kwargs: object())

    assert server._notif_submit("req", "sid", _claimed_session(), "notify", "test") is accepted
    assert frames == ([("message.start", "sid")] if accepted else [])
