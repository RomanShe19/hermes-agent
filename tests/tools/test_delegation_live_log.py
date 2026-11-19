"""Tests for tools/delegation_live_log.py — live subagent transcripts.

Covers:
- writer event rendering + truncation + append/flush semantics
- failure-swallowing when the target dir is unwritable
- the tool_progress_callback observe() demux (assistant/tool events in order)
- dispatch-time creation: paths pre-created with a header, manifest written
- retention pruning of stale live dirs
- delegate_task return-shape: live_transcripts in sync + background dispatch
"""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import delegation_live_log as dll
from tools.delegation_live_log import (
    LiveTranscriptWriter,
    create_live_transcripts,
    live_transcript_root,
    prune_stale_live_dirs,
    update_manifest_statuses,
    wrap_progress_callback,
)


# ---------------------------------------------------------------------------
# Writer unit tests
# ---------------------------------------------------------------------------


def test_writer_precreates_file_with_header():
    w = LiveTranscriptWriter("deleg_test1", 0, "do the thing", context="some ctx")
    assert w.path is not None and w.path.exists()
    text = w.path.read_text(encoding="utf-8")
    assert "Hermes subagent live transcript" in text
    assert "delegation: deleg_test1" in text
    assert "goal: do the thing" in text
    assert "kickoff" in text
    assert "some ctx" in text
    # Lives under the hermes cache/delegation/live root, named task-<n>.log
    assert w.path.name == "task-0.log"
    assert w.path.parent.name == "deleg_test1"
    assert w.path.parent.parent == live_transcript_root()


def test_stream_deltas_buffer_and_flush_as_one_line():
    w = LiveTranscriptWriter("deleg_stream", 0, "g")
    w.add_stream_delta("Hello ")
    w.add_stream_delta("world, ")
    w.add_stream_delta("streaming.")
    # Not yet flushed
    assert "Hello world" not in w.path.read_text(encoding="utf-8")
    w.flush_stream()
    text = w.path.read_text(encoding="utf-8")
    assert "Hello world, streaming." in text
    # tool_start also flushes pending stream text first
    w.add_stream_delta("more text")
    w.tool_start("read_file", "foo.py")
    text = w.path.read_text(encoding="utf-8")
    assert text.index("more text") < text.index("-> read_file")


# ---------------------------------------------------------------------------
# observe() demux — the tool_progress_callback seam
# ---------------------------------------------------------------------------


def test_observe_maps_child_callback_events_to_lines():
    w = LiveTranscriptWriter("deleg_observe", 0, "g")
    w.observe("subagent.start", preview="kick off the goal")
    w.observe("_thinking", "first line of thinking")
    w.observe("reasoning.available", "_thinking", "deep reasoning text", None)
    w.observe("tool.started", "terminal", "ls /tmp", {"command": "ls /tmp"})
    w.observe("tool.completed", "terminal", None, None,
              duration=0.5, is_error=False, result="ok output")
    w.observe("subagent.text", preview="final reply ")
    w.observe("subagent.text", preview="streamed in parts")
    w.observe("subagent.complete", preview="short", status="completed",
              duration_seconds=3.2, summary="did the thing")
    text = w.path.read_text(encoding="utf-8")
    assert "kick off the goal" in text
    assert "first line of thinking" in text
    assert "deep reasoning text" in text
    assert "-> terminal(ls /tmp)" in text
    assert "terminal ok 0.5s: ok output" in text
    assert "final reply streamed in parts" in text
    assert "status=completed" in text
    assert "did the thing" in text


def test_finalize_records_budget_exhaustion_and_errors():
    w = LiveTranscriptWriter("deleg_final", 0, "g")
    w.finalize({"status": "failed", "exit_reason": "max_iterations",
                "error": "Subagent did not produce a response."})
    text = w.path.read_text(encoding="utf-8")
    assert "end status=failed" in text
    assert "exit_reason=max_iterations" in text
    assert "iteration budget exhausted" in text
    assert "did not produce a response" in text


def test_wrap_progress_callback_tees_and_preserves_inner():
    w = LiveTranscriptWriter("deleg_wrap", 0, "g")
    seen = []

    def inner(event_type, tool_name=None, preview=None, args=None, **kw):
        seen.append((event_type, tool_name))

    inner_flushed = []
    inner._flush = lambda: inner_flushed.append(True)

    cb = wrap_progress_callback(inner, w)
    cb("tool.started", "terminal", "echo hi", None)
    cb("_thinking", "pondering")
    assert seen == [("tool.started", "terminal"), ("_thinking", "pondering")]
    text = w.path.read_text(encoding="utf-8")
    assert "-> terminal(echo hi)" in text and "pondering" in text
    # _flush contract preserved
    cb._flush()
    assert inner_flushed == [True]


def test_wrap_progress_callback_writer_failure_does_not_block_inner():
    w = LiveTranscriptWriter("deleg_wfail", 0, "g")
    w.observe = MagicMock(side_effect=RuntimeError("disk on fire"))
    seen = []
    cb = wrap_progress_callback(lambda *a, **k: seen.append(a), w)
    cb("tool.started", "terminal", "x", None)  # must not raise
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Dispatch-time creation + manifest + retention
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# delegate_task return-shape integration
# ---------------------------------------------------------------------------


def _make_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess-live"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    return parent


_CREDS = {
    "model": "m", "provider": None, "base_url": None, "api_key": None,
    "api_mode": None, "command": None, "args": None,
}


def _fake_run(task_index, goal, child=None, parent_agent=None, **kw):
    return {
        "task_index": task_index, "status": "completed",
        "summary": f"done: {goal}", "api_calls": 1,
        "duration_seconds": 0.1, "model": "m", "exit_reason": "completed",
    }


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Credential redaction
# ---------------------------------------------------------------------------
#
# These transcripts land under ``cache/delegation``, which delegate_tool mounts
# READ-ONLY into remote terminal backends — so a line written here is readable
# from inside the sandbox. The rendered events are exactly the secret-bearing
# surfaces (tool args, tool results, streamed assistant text), and every other
# sink for that data already routes through the canonical redactor.

_BEARER = "sk-ant-api03-" + "R" * 24
_ENV_KEY = "sk-proj-" + "L" * 24
_AWS = "wJalrXUtnFEMIK7MDENG" + "bPxRfiCY"


def test_tool_args_are_redacted_before_hitting_disk():
    w = LiveTranscriptWriter("deleg_redact_args", 0, "g")
    w.observe(
        "tool.started",
        "terminal",
        f'curl -H "Authorization: Bearer {_BEARER}" https://api.internal',
        None,
    )
    body = w.path.read_text(encoding="utf-8")
    assert _BEARER not in body
    assert "terminal" in body, "redaction must not gut the operational detail"


def test_tool_results_are_redacted_before_hitting_disk():
    w = LiveTranscriptWriter("deleg_redact_result", 0, "g")
    w.observe(
        "tool.completed",
        "terminal",
        None,
        None,
        result=f"OPENAI_API_KEY={_ENV_KEY}\nAWS_SECRET_ACCESS_KEY={_AWS}",
        duration=0.4,
    )
    body = w.path.read_text(encoding="utf-8")
    assert _ENV_KEY not in body
    assert _AWS not in body
    assert "OPENAI_API_KEY" in body, "key NAMES stay — only the values are masked"


def test_streamed_assistant_text_is_redacted():
    w = LiveTranscriptWriter("deleg_redact_stream", 0, "g")
    w.observe("subagent.text", None, f"the key is {_ENV_KEY}")
    w.flush_stream()
    assert _ENV_KEY not in w.path.read_text(encoding="utf-8")


def test_goal_header_is_redacted():
    """The header bypasses event(); a pasted key in the goal must not survive."""
    w = LiveTranscriptWriter("deleg_redact_goal", 0, f"deploy using {_BEARER}")
    body = w.path.read_text(encoding="utf-8")
    assert _BEARER not in body
    assert "deploy using" in body


def test_manifest_goal_is_redacted():
    """manifest.json shares the mounted dir with the .log files.

    Redacting the log header while ``_write_manifest`` serialises the same goal
    verbatim would leave the credential exposed one file over — both sinks in
    ``cache/delegation/live/<id>/`` are readable from inside a sandbox.
    """
    delegation_id, _writers, _paths = create_live_transcripts(
        [{"goal": f"deploy using {_BEARER}"}]
    )

    manifest = json.loads(
        (live_transcript_root() / delegation_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    goal = manifest["tasks"][0]["goal"]

    assert _BEARER not in goal
    assert "deploy using" in goal, "redaction must not blank the goal entirely"


# ---------------------------------------------------------------------------
# Explicit profile home (#91996)
# ---------------------------------------------------------------------------

def test_explicit_home_pins_transcripts_across_raw_thread_boundary(tmp_path, monkeypatch):
    """Ambient resolve falls through to process HERMES_HOME when the session's
    ContextVar override is dropped by a raw threading.Thread boundary; an
    explicit home keeps transcripts + manifest under the originating profile."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    process_home = tmp_path / "process-default"
    session_home = tmp_path / "session-profile"
    process_home.mkdir()
    session_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(process_home))
    token = set_hermes_home_override(session_home)
    out = {}
    try:
        def worker():
            _id, _writers, paths = create_live_transcripts(
                [{"goal": "profile-pinned repro"}], home=session_home,
            )
            out["paths"] = paths

        t = threading.Thread(target=worker)
        t.start()
        t.join()
    finally:
        reset_hermes_home_override(token)

    expected_root = session_home / "cache" / "delegation" / "live"
    assert out["paths"], "transcripts were created"
    for p in out["paths"]:
        assert Path(p).is_relative_to(expected_root), p
    deleg_dir = Path(out["paths"][0]).parent
    manifest = json.loads((deleg_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_count"] == 1


def test_manifest_update_uses_same_explicit_home(tmp_path, monkeypatch):
    """update_manifest_statuses must resolve the manifest through the same
    explicit home create used, or the status write is silently lost."""
    ambient_home = tmp_path / "other-profile"
    ambient_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))

    home = tmp_path / "own-profile"
    home.mkdir()

    deleg_id, _writers, paths = create_live_transcripts(
        [{"goal": "pin me"}], home=home,
    )
    update_manifest_statuses(
        deleg_id,
        [{"task_index": 0, "status": "completed"}],
        home=home,
    )
    manifest = json.loads(
        (Path(paths[0]).parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tasks"][0]["status"] == "completed"


def test_parent_without_session_db_warns_and_skips_pin(caplog):
    """A parent with no _session_db cannot be pinned; the skip must be
    observable (ambient fallback is the #91996 failure mode), not silent."""
    import logging as _logging

    from tools.delegate_tool import _parent_live_home

    parent = MagicMock(spec=[])  # no _session_db attribute at all

    with caplog.at_level(_logging.WARNING, logger="tools.delegate_tool"):
        assert _parent_live_home(parent) is None
    assert any("home pinning skipped" in r.getMessage() for r in caplog.records), (
        "the ambient fallback must leave a warning trace"
    )


def test_parent_session_db_pins_home():
    from tools.delegate_tool import _parent_live_home

    parent = MagicMock(spec=["_session_db"])
    parent._session_db.db_path = Path("/prof/state.db")

    assert _parent_live_home(parent) == Path("/prof")


def test_mock_parent_db_path_is_not_pinned(caplog):
    """Regression (PR #131931 side-effect screen): a duck-typed parent whose
    ``_session_db.db_path`` is auto-created by a Mock is not a real path —
    pinning to it would send transcripts to ``str(<MagicMock>)`` paths.  The
    pin must fall back to None (ambient resolve), and the skip must leave a
    trace.  Note the check is for concrete ``str``/``Path``, NOT the
    ``os.PathLike`` protocol: MagicMock registers ``__fspath__`` and so IS
    PathLike, which is exactly how the garbage pin slipped through."""
    import logging as _logging

    from tools.delegate_tool import _parent_live_home

    parent = MagicMock()  # plain mock: _session_db.db_path auto-creates a Mock

    with caplog.at_level(_logging.DEBUG, logger="tools.delegate_tool"):
        assert _parent_live_home(parent) is None
    assert any("home pinning skipped" in r.getMessage() for r in caplog.records), (
        "the skipped pin must leave a log trace"
    )


def test_parent_db_path_accepts_str_paths():
    """SessionDB always holds a Path, but a str db_path is a valid path value
    too — it must pin, not be rejected by the concrete-type guard."""
    from tools.delegate_tool import _parent_live_home

    parent = MagicMock(spec=["_session_db"])
    parent._session_db.db_path = "/prof/state.db"

    assert _parent_live_home(parent) == Path("/prof")


def test_prune_sweeps_the_pinned_root_not_ambient(tmp_path, monkeypatch):
    """Retention must clean where the dispatch writes: stale dirs under the
    pinned home are removed by that home's dispatches; a stale dir under the
    ambient root is NOT this dispatch's business."""
    ambient_home = tmp_path / "ambient"
    pinned_home = tmp_path / "pinned"
    ambient_home.mkdir()
    pinned_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))

    stale_age = time.time() - (dll.LIVE_RETENTION_DAYS + 1) * 86400
    stale_pinned = pinned_home / "cache" / "delegation" / "live" / "old-deleg"
    stale_pinned.mkdir(parents=True)
    os.utime(stale_pinned, (stale_age, stale_age))
    stale_ambient = (
        ambient_home / "cache" / "delegation" / "live" / "other-profile-deleg"
    )
    stale_ambient.mkdir(parents=True)
    os.utime(stale_ambient, (stale_age, stale_age))

    create_live_transcripts([{"goal": "sweep my own root"}], home=pinned_home)

    assert not stale_pinned.exists(), "pinned home's own stale dir must be pruned"
    assert stale_ambient.exists(), "ambient root is not this dispatch's root"


def test_manifest_includes_model_and_provider():
    """The manifest.json should record the model and provider used for the delegation."""
    delegation_id, _writers, _paths = create_live_transcripts(
        [{"goal": "task 1"}, {"goal": "task 2"}],
        model="openrouter/gpt-4o",
        provider="openrouter",
    )

    manifest = json.loads(
        (live_transcript_root() / delegation_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["model"] == "openrouter/gpt-4o"
    assert manifest["provider"] == "openrouter"
    # tasks array should not be affected
    assert len(manifest["tasks"]) == 2




def test_no_file_in_the_dispatch_directory_carries_the_raw_key():
    """Whole-directory sweep: every artefact dispatch writes is covered."""
    delegation_id, _writers, _paths = create_live_transcripts(
        [{"goal": f"deploy using {_BEARER}"}, {"goal": "second task"}]
    )

    directory = live_transcript_root() / delegation_id
    written = sorted(p.name for p in directory.iterdir())

    assert "manifest.json" in written
    assert any(name.endswith(".log") for name in written)
    for path in directory.iterdir():
        assert _BEARER not in path.read_text(encoding="utf-8"), (
            f"{path.name} leaked the credential"
        )


def test_thinking_text_is_redacted():
    w = LiveTranscriptWriter("deleg_redact_think", 0, "g")
    w.observe("_thinking", f"I should use {_ENV_KEY} here")
    assert _ENV_KEY not in w.path.read_text(encoding="utf-8")


def test_redaction_covers_every_helper_via_the_event_chokepoint():
    """Any helper that reaches disk goes through event(), so all are covered."""
    w = LiveTranscriptWriter("deleg_redact_all", 0, "g")
    w.assistant_text(f"a {_ENV_KEY}")
    w.thinking(f"b {_ENV_KEY}")
    w.tool_start("terminal", f"c {_ENV_KEY}")
    w.tool_result("terminal", result=f"d {_ENV_KEY}")
    w.marker(f"e {_ENV_KEY}")
    w.finalize({"status": "error", "error": f"f {_ENV_KEY}"})
    body = w.path.read_text(encoding="utf-8")
    assert _ENV_KEY not in body, "a write path escaped the redactor"


def test_benign_transcript_content_is_untouched():
    """Redaction must not mangle ordinary transcript text."""
    w = LiveTranscriptWriter("deleg_redact_benign", 0, "refactor the parser")
    w.observe("tool.started", "read_file", "src/parser.py", None)
    w.observe("tool.completed", "read_file", None, None, result="def parse(x): ...", duration=1.5)
    body = w.path.read_text(encoding="utf-8")
    assert "src/parser.py" in body
    assert "def parse(x)" in body
    assert "refactor the parser" in body


def test_manifest_completes_only_when_all_units_finish():
    delegation_id, _writers, paths = create_live_transcripts([{"goal": "first"}, {"goal": "second"}])
    manifest_path = Path(paths[0]).parent / "manifest.json"

    update_manifest_statuses(delegation_id, [{"task_index": 0, "status": "completed", "schema_valid": False}])
    partial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "completed" not in partial
    assert [task["status"] for task in partial["tasks"]] == ["completed", "running"]
    assert partial["tasks"][0]["schema_valid"] is False

    update_manifest_statuses(delegation_id, [{"task_index": 1, "status": "failed", "exit_reason": "error"}])
    done = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert done["completed"] and [task["status"] for task in done["tasks"]] == ["completed", "failed"]
    assert done["tasks"][0]["schema_valid"] is False
    assert done["tasks"][1]["exit_reason"] == "error"


def test_concurrent_units_do_not_overwrite_manifest_statuses(monkeypatch):
    import threading
    from tools import delegation_live_log as log

    delegation_id, _writers, paths = create_live_transcripts([{"goal": "first"}, {"goal": "second"}])
    entered, first_can_finish, second_written = threading.Event(), threading.Event(), threading.Event()
    dump = log._dump_json

    def pause_first_write(path, manifest):
        if threading.current_thread().name == "first-unit":
            entered.set()
            assert first_can_finish.wait(5)
        dump(path, manifest)
        if threading.current_thread().name == "second-unit":
            second_written.set()

    monkeypatch.setattr(log, "_dump_json", pause_first_write)
    first, second = (
        threading.Thread(
            name=name, target=update_manifest_statuses,
            args=(delegation_id, [{"task_index": index, "status": "completed"}]),
        )
        for index, name in enumerate(("first-unit", "second-unit"))
    )
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        second_written.wait(0.7)  # With no lock, the second unit overwrites the same pre-update manifest.
    finally:
        first_can_finish.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)

    assert not first.is_alive() and not second.is_alive()
    manifest = json.loads((Path(paths[0]).parent / "manifest.json").read_text(encoding="utf-8"))
    assert [task["status"] for task in manifest["tasks"]] == ["completed", "completed"]


def test_missing_transcript_keeps_existing_log_with_its_task(monkeypatch):
    from tools import delegate_tool as dt
    from tools.delegate_tool_dispatch import _dispatched_payload

    batches = []
    run_batch = dt._run_batch

    def capture(batch, background):
        batches.append(batch)
        return run_batch(batch, background)

    monkeypatch.setattr(dt, "_run_batch", capture)

    write = Path.write_text

    def fail_first_log(path, *args, **kwargs):
        if path.name == "task-0.log":
            raise OSError("test: first transcript unavailable")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_first_log)
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kwargs: MagicMock())
    monkeypatch.setattr(dt, "_run_single_child", _fake_run)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *args: _CREDS)
    result = json.loads(dt.delegate_task(
        tasks=[{"goal": "Check first input"}, {"goal": "Check second input"}], parent_agent=_make_parent(), background=False,
    ))

    assert "results" in result, result
    [second_path] = result["live_transcripts"]
    assert Path(second_path).name == "task-1.log"
    assert "live_transcript" not in result["results"][0]
    assert result["results"][1]["live_transcript"] == second_path
    manifest = json.loads((Path(second_path).parent / "manifest.json").read_text(encoding="utf-8"))
    assert [task["log"] for task in manifest["tasks"]] == [None, second_path]

    [batch] = batches
    payload = _dispatched_payload(batch, [(batch, batch.live_deleg_id)])
    assert payload["live_transcripts"] == [second_path]
    assert payload["task_transcripts"] == {"1": second_path}


def test_child_preflight_failure_closes_unstarted_siblings_and_finalizes_logs(monkeypatch):
    from tools import delegate_tool as dt

    parent = _make_parent()
    parent._active_children = []
    first = MagicMock()
    progress = MagicMock()
    first.tool_progress_callback = progress

    def build_child(**kwargs):
        if kwargs["task_index"] == 1:
            raise ValueError("worker preflight refused")
        parent._active_children.append(first)
        return first

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", build_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *args: _CREDS)
    monkeypatch.setattr(dt, "_run_batch", lambda *a: pytest.fail("rejected batch must not start"))

    result = json.loads(dt.delegate_task(
        tasks=[{"goal": "Prepare first step"}, {"goal": "Rejected step"}, {"goal": "Remaining step"}],
        parent_agent=parent,
    ))

    assert result["error"] == "worker preflight refused"
    assert parent._active_children == []
    first.close.assert_called_once_with()
    progress.assert_called_once_with("subagent.complete", None,
                                     "Batch not started after worker preflight failed.", None, status="failed")
    [manifest_path] = live_transcript_root().glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [task["status"] for task in manifest["tasks"]] == ["rejected", "error", "rejected"]
    assert manifest["completed"]
    assert "end status=error" in (manifest_path.parent / "task-1.log").read_text(encoding="utf-8")


def test_group_completion_keeps_only_its_owned_task_context_after_replay(monkeypatch):
    import queue
    from dataclasses import replace
    from tools import async_delegation as ledger, delegate_tool as dt
    from tools.delegate_tool_dispatch import _dispatch_unit
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import format_process_notification

    batches = []
    monkeypatch.setattr(dt, "_run_batch", lambda batch, background: batches.append(batch) or "{}")
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kwargs: MagicMock())
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *args: _CREDS)
    monkeypatch.setattr(dt, "_run_single_child", _fake_run)
    events = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", events)
    ledger._reset_for_tests()

    protected_context = "Verify backup only. Do not modify the other owner's container."
    unrelated_context = "Review the format of the unrelated report."
    dt.delegate_task(
        context="Release verification.",
        tasks=[{"goal": "Verify the source backup", "context": f"{protected_context} {_BEARER}"},
               {"goal": "Review the report format", "context": unrelated_context}],
        parent_agent=_make_parent(), background=True,
    )
    [batch] = batches
    unit = replace(batch, children=batch.children[:1], group="backup")
    try:
        receipt = _dispatch_unit(unit, "deleg-context", None, {
            "session_key": "owner-session", "origin_ui_session_id": "owner-ui", "origin_session_id": "owner-durable",
            "parent_session_id": "owner-agent", "max_async_children": 1,
        })
        assert receipt["status"] == "dispatched"
        live = events.get(timeout=5)
        assert live["delegation_id"] == receipt["delegation_id"]
        assert set(live["task_contexts"]) == {"0"}
        assert protected_context in live["task_contexts"]["0"]
        assert _BEARER not in live["task_contexts"]["0"]

        ledger.return_completion_offer(live)
        assert ledger.restore_undelivered_completions(events) == 1
        replay = events.get(timeout=5)
        assert replay["restored"] is True
        text = format_process_notification(replay)
        assert "Release verification." in text
        assert protected_context in text
        assert unrelated_context not in text
        assert _BEARER not in text
    finally:
        ledger._reset_for_tests()
