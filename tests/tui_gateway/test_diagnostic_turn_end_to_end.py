"""Diagnostic wake exercises the actual TUI turn, event transport, and control callback."""
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tests.tui_gateway.test_auto_continue import turn_env, marker_home, _session


@pytest.mark.parametrize("suppress", [False, True])
def test_diagnostic_turn_runs_but_never_echoes_on_wire(turn_env, marker_home, monkeypatch, suppress):
    monkeypatch.setenv("HERMES_HOME", str(marker_home))
    (marker_home / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(not suppress).lower()}}}")
    owner = marker_home / "owner"
    owner.mkdir()
    (owner / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}")
    frames, work, controls = [], [], []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (threading.Event(), SimpleNamespace(join=lambda: None)))
    agent = SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
                            clarify_callback=lambda *a: controls.append(a))
    def run(message, **kwargs):
        work.append(message)
        kwargs["stream_callback"]("technical diagnostic echo")
        if agent.interim_assistant_callback:
            agent.interim_assistant_callback("technical interim")
        agent.clarify_callback("Recovery approval?", ["yes", "no"])
        return {"final_response": "technical final", "messages": []}
    agent.run_conversation = run
    session = _session(agent=agent, running=True)
    session["profile_home"] = str(owner)
    server._run_prompt_submit("request", "session", session, "engine failure",
                              display_metadata={"notification_category": "diagnostic"})
    assert work == ["engine failure"]
    assert controls == [("Recovery approval?", ["yes", "no"])]
    content_frames = [f for f in frames if (f.get("params") or {}).get("type") in {"message.delta", "message.interim", "message.complete", "error"}]
    assert bool(content_frames) is not suppress
    starts = [f for f in frames if (f.get("params") or {}).get("type") == "message.start"]
    assert len(starts) == (0 if suppress else 1)
    if not suppress:
        assert frames.index(starts[0]) < frames.index(content_frames[0])
    assert session["running"] is False


def test_restart_preserves_notification_input_policy_through_marker(turn_env, marker_home, monkeypatch):
    from tui_gateway.turn_marker import read_turn_marker
    session = _session()
    server._record_turn_marker(session, "engine diagnostic",
                               notification_category="diagnostic", resolve_context_refs=False)
    marker = read_turn_marker(marker_home, "session-key")
    assert marker["notification_category"] == "diagnostic"
    calls = []
    monkeypatch.setattr(server, "_start_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(server, "_auto_continue_config", lambda: (True, 3600, 2))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: calls.append(k))
    server._maybe_schedule_auto_continue("session", session, "session-key")
    assert calls[0]["display_metadata"]["notification_category"] == "diagnostic"
    assert calls[0]["resolve_context_refs"] is False


def test_next_human_followup_is_outside_diagnostic_presentation_scope(turn_env, marker_home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(marker_home))
    (marker_home / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (threading.Event(), SimpleNamespace(join=lambda: None)))
    agent = SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
        run_conversation=lambda *a, **k: {"final_response": "diagnostic echo", "messages": []})
    session = _session(agent=agent, running=True)
    def followups(*a):
        server._emit("message.complete", "session", {"text": "next requested result"})
    monkeypatch.setattr(server, "_run_post_turn_followups", followups)
    server._run_prompt_submit("request", "session", session, "engine failure",
                              display_metadata={"notification_category": "diagnostic"})
    finals = [f["params"]["payload"]["text"] for f in frames if f.get("params", {}).get("type") == "message.complete"]
    assert finals == ["next requested result"]

def test_delegation_completion_reference_is_literal(turn_env, marker_home, monkeypatch):
    from agent import context_references
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import async_delegation as ledger

    invoked = []
    references = []

    def refuse_reference(text, **kwargs):
        references.append(text)
        return SimpleNamespace(blocked=True, warnings=["Context injection refused."])

    def run(text, **kwargs):
        invoked.append(text)
        return {"final_response": "processed"}

    monkeypatch.setattr(context_references, "preprocess_context_references", refuse_reference)
    monkeypatch.setattr(server, "_emit", lambda *args: None)
    session = _session(agent=SimpleNamespace(session_id="session-key", run_conversation=run,
                                            clear_interrupt=lambda: None), running=True)
    event = {"type": "async_delegation", "delegation_id": "literal-reference",
             "session_key": "session-key", "status": "completed", "dispatched_at": 1.0}
    text = "Worker result includes the literal @diff"
    token = set_hermes_home_override(str(marker_home))
    try:
        ledger._persist_dispatch(event)
        ledger._persist_completion(event, {"status": "completed", "summary": text})
        assert server._notif_dispatch_event("sid", session, event, text)
        assert ledger.get_durable_delegation(event["delegation_id"])["delivery_state"] == "delivered"
    finally:
        reset_hermes_home_override(token)

    assert not references
    assert len(invoked) == 1 and text in invoked[0]


@pytest.mark.parametrize("failure_phase", ["before_ack", "queued", "claim_lost", "model_invoked"])
def test_only_unprepared_completion_can_replay(turn_env, marker_home, monkeypatch, failure_phase):
    import queue
    import time
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import async_delegation as ledger

    frames, scheduled = [], []
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload=None: frames.append((kind, payload)))
    if failure_phase == "model_invoked":
        def fail_after_start(*args):
            raise RuntimeError("failure after entering the model path")

        monkeypatch.setattr(server, "_invoke_agent", fail_after_start)
    elif failure_phase == "claim_lost":
        monkeypatch.setattr(ledger, "complete_event_delivery", lambda *args: False)
        monkeypatch.setattr(server, "_invoke_agent", lambda *args: pytest.fail("lost claim ran the model"))
    else:
        monkeypatch.setattr(server, "_prepare_turn_input", lambda *args: None)
    if failure_phase == "queued":
        monkeypatch.setattr(server, "_start_session_work",
                            lambda target, **kwargs: scheduled.append(target) or object())
    agent = SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
                            run_conversation=lambda *a, **k: pytest.fail("unprepared input ran"))
    session = _session(agent=agent, running=True)
    event = {"type": "async_delegation", "delegation_id": "preflight-replay",
             "session_key": "session-key", "status": "completed", "dispatched_at": time.time()}
    original = {"status": "completed", "summary": "retain the durable result"}
    token = set_hermes_home_override(str(marker_home))
    try:
        ledger._persist_dispatch(event)
        ledger._persist_completion(event, original)
        assert server._notif_dispatch_event("sid", session, event, "result")
        if failure_phase == "queued":
            [target] = scheduled
            row = ledger.get_durable_delegation(event["delegation_id"])
            assert row["delivery_state"] == "pending"
            assert row["delivery_attempts"] == 1
            target()

        row = ledger.get_durable_delegation(event["delegation_id"])
        assert row["result"] == original
        assert session["running"] is False
        terminal = [payload for kind, payload in frames if kind == "message.complete"]
        q = queue.Queue()
        replayed = ledger.sweep_orphaned_completions(q, now=time.time() + ledger._ORPHAN_STALE_S + 1)
        if failure_phase == "model_invoked":
            assert row["delivery_state"] == "delivered"
            assert [payload["status"] for payload in terminal] == ["error"]
            assert replayed == 0
        else:
            assert terminal == []
            assert not any(kind == "message.start" for kind, _ in frames)
            assert row["delivery_state"] == "pending"
            assert replayed == 1
            assert q.get_nowait()["delegation_id"] == event["delegation_id"]
    finally:
        reset_hermes_home_override(token)


def test_legacy_completion_without_ledger_still_runs(turn_env, marker_home, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    invoked = []
    monkeypatch.setattr(server, "_invoke_agent", lambda *args: invoked.append(args))
    event = {"type": "async_delegation", "delegation_id": "legacy", "status": "completed"}
    token = set_hermes_home_override(str(marker_home))
    try:
        assert server._notif_dispatch_event("sid", _session(running=True), event, "legacy result")
    finally:
        reset_hermes_home_override(token)
    assert len(invoked) == 1
