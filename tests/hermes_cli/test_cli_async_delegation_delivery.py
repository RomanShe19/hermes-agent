"""Regression coverage for CLI async-delegation completion ownership."""

import queue
from types import SimpleNamespace

from cli import HermesCLI


def test_cli_completion_drain_uses_visible_session_identity(monkeypatch):
    """A CLI window must not claim another window's restored completion."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()

    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_visible",
        "session_key": "visible-session",
    }
    calls = []

    class FakeRegistry:
        completion_queue = queue.Queue()

        def drain_notifications(self, *, session_key="", owns_event=None):
            calls.append((session_key, owns_event(event)))
            return [(event, "completion payload")]

    claimed = []
    completed = []

    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        FakeRegistry(),
    )
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: claimed.append((evt, consumer)) or "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._drain_process_notifications("cli-idle")

    assert calls == [("visible-session", True)]
    assert cli._pending_input.get_nowait() == "completion payload"
    assert claimed == []
    assert completed == []


def test_cli_queued_completion_is_pending_until_a_turn_can_start(monkeypatch, tmp_path):
    import time
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import async_delegation as ledger

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "owner"
    cli._pending_input = queue.Queue()
    completed_at = time.time()
    event = {"type": "async_delegation", "delegation_id": "deleg-cli-pending",
             "session_key": "owner", "status": "completed", "completed_at": completed_at}

    class Registry:
        completion_queue = queue.Queue()

        def drain_notifications(self, *, session_key, owns_event):
            return [(event, "worker result")] if owns_event(event) else []

    registry = Registry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    token = set_hermes_home_override(str(tmp_path))
    try:
        ledger._persist_dispatch(dict(event, dispatched_at=completed_at - 1))
        ledger._persist_completion(event, {"status": "completed"})
        cli._drain_process_notifications("cli-idle")
        notification = cli._pending_input.get_nowait()
        assert ledger.get_durable_delegation(event["delegation_id"])["delivery_state"] == "pending"

        # An actual credential refusal in chat() must leave the row retryable.
        cli._secret_capture_callback = None
        monkeypatch.setattr("cli.set_secret_capture_callback", lambda callback: None)
        cli._ensure_runtime_credentials = lambda: False
        cli._typed_voice_stop = lambda text: False
        cli._pending_resume_sessions = []
        cli.handle_bang_shell = lambda text: False
        cli._print_user_message_preview = lambda text: None
        cli._turn_summary_begin = lambda: None
        cli._app = SimpleNamespace(invalidate=lambda: None)
        cli._tui_after_turn = lambda: None
        assert cli._tui_process_one_input(notification) is None
        assert ledger.get_durable_delegation(event["delegation_id"])["delivery_state"] == "pending"

        retry_at = time.time() + ledger._ORPHAN_STALE_S + 1
        assert ledger.sweep_orphaned_completions(registry.completion_queue, now=retry_at) == 1
        replay = registry.completion_queue.get_nowait()
        assert replay["restored"] and replay["delegation_id"] == event["delegation_id"]

        cli.session_id = "new-session"
        assert cli._tui_process_one_input(notification) is None
        assert registry.completion_queue.get_nowait() == event
        assert ledger.get_durable_delegation(event["delegation_id"])["delivery_state"] == "pending"

        class StartedThread:
            def __init__(self, *, target, args, daemon):
                assert args[1] == "worker result"

            def start(self):
                assert ledger.get_durable_delegation(event["delegation_id"])["delivery_state"] == "pending"

        from hermes_cli import cli_chat_turn_mixin

        cli.session_id = "owner"
        cli._ensure_runtime_credentials = lambda: True
        cli._active_agent_route_signature = "unchanged"
        cli._resolve_turn_agent_config = lambda message: {
            "signature": "unchanged", "model": "synthetic", "runtime": None,
        }
        cli.agent = SimpleNamespace()
        cli.conversation_history = []
        cli._init_agent = lambda **kwargs: True
        cli._sync_fallback_chain_with_config = lambda agent: None
        cli._reset_stream_state = lambda: None
        cli._chat_setup_turn_audio = lambda *args: None
        cli._chat_monitor_agent_thread = lambda *args: None
        cli._chat_settle_turn = lambda turn: None
        cli._chat_render_turn = lambda *args: "continued"
        cli._chat_release_turn_audio = lambda turn: None
        monkeypatch.setattr(cli_chat_turn_mixin, "threading", SimpleNamespace(Thread=StartedThread))
        monkeypatch.setattr("cli.ChatConsole", lambda: SimpleNamespace(print=lambda *args: None))
        monkeypatch.setattr("cli._accent_hex", lambda: "ffffff")
        assert cli._tui_process_one_input(notification) == "continued"
        assert [message["content"] for message in cli.conversation_history] == ["worker result"]
        assert ledger.get_durable_delegation(event["delegation_id"])["delivery_state"] == "delivered"
        assert ledger.claim_event_delivery(event, "duplicate") is None
    finally:
        reset_hermes_home_override(token)


def test_cli_completion_ownership_rejects_foreign_session():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = None

    assert not cli._owns_process_notification(
        {"type": "async_delegation", "session_key": "foreign-session"}
    )


def test_cli_completion_ownership_accepts_compression_lineage():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"

    class FakeSessionDB:
        def resolve_resume_session_id(self, session_id):
            assert session_id == "pre-compression-session"
            return "visible-session"

    cli._session_db = FakeSessionDB()

    assert cli._owns_process_notification(
        {
            "type": "async_delegation",
            "session_key": "pre-compression-session",
        }
    )
