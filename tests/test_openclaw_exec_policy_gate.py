"""Daemon drives OpenClaw's native exec-approval gate from the active policies.

ClawMetry's own watcher is reactive (can't prevent a command). When a
require-approval policy covering exec is active, the daemon applies
`openclaw exec-policy preset cautious` (pre-execution gate); when none are,
it restores `yolo` — but only if it was the one that set cautious, so a
hand-set posture is never clobbered. No-op off an OpenClaw host.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clawmetry import approvals  # noqa: E402


def _wire(monkeypatch, tmp_path, has_openclaw=True, prev=None):
    state = tmp_path / "exec_policy_applied"
    if prev is not None:
        state.write_text(prev)
    monkeypatch.setattr(approvals, "_EXEC_POLICY_STATE", state)
    monkeypatch.setattr(approvals, "_EXEC_POLICY_BACKOFF",
                        {"fails": 0, "until": 0.0})
    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: (("/usr/local/bin/openclaw" if has_openclaw else None), {}))
    applied = []
    monkeypatch.setattr(approvals, "_apply_openclaw_exec_preset",
                        lambda preset: (applied.append(preset), True)[1])
    return applied, state


RM = {"action": "require_approval", "tool": "exec",
      "pattern": r"rm\s+-rf", "enabled": True}
SECRETS = {"action": "require_approval", "tool": "",
           "pattern": r"\.env", "enabled": True}


def test_enable_applies_cautious(monkeypatch, tmp_path):
    applied, state = _wire(monkeypatch, tmp_path)
    approvals.sync_openclaw_exec_policy([RM])
    assert applied == ["cautious"]
    assert state.read_text() == "cautious"


def test_secrets_rule_tool_agnostic_still_gates(monkeypatch, tmp_path):
    applied, _ = _wire(monkeypatch, tmp_path)
    approvals.sync_openclaw_exec_policy([SECRETS])
    assert applied == ["cautious"]


def test_no_reapply_when_already_cautious(monkeypatch, tmp_path):
    applied, _ = _wire(monkeypatch, tmp_path, prev="cautious")
    approvals.sync_openclaw_exec_policy([RM])
    assert applied == []  # idempotent — desired == last applied


def test_disable_restores_yolo_only_if_we_set_cautious(monkeypatch, tmp_path):
    applied, state = _wire(monkeypatch, tmp_path, prev="cautious")
    approvals.sync_openclaw_exec_policy([])  # all rules off
    assert applied == ["yolo"]
    assert state.read_text() == "yolo"


def test_disable_does_not_force_yolo_on_handset_posture(monkeypatch, tmp_path):
    # No prior state (operator may have set deny-all by hand) → never relax.
    applied, _ = _wire(monkeypatch, tmp_path, prev=None)
    approvals.sync_openclaw_exec_policy([])
    assert applied == []


def test_noop_off_openclaw_host(monkeypatch, tmp_path):
    applied, _ = _wire(monkeypatch, tmp_path, has_openclaw=False)
    approvals.sync_openclaw_exec_policy([RM])
    assert applied == []


def test_disabled_policy_does_not_gate(monkeypatch, tmp_path):
    applied, _ = _wire(monkeypatch, tmp_path)
    # A cloud policy row that is present but disabled must not trigger the gate.
    # (load_policies already filters enabled; _policies_want_exec_gate keys off
    # action/tool, so we assert the want-detector directly for a non-exec tool.)
    approvals.sync_openclaw_exec_policy([{"action": "require_approval",
                                          "tool": "browser", "enabled": True}])
    assert applied == []  # browser-only rule doesn't imply an exec gate


def test_failed_apply_backs_off_then_retries(monkeypatch, tmp_path):
    # A host that can't complete the CLI (live-hit 2026-07-10: node child
    # outliving the timeout on a 1-core box) must not retry every watcher
    # iteration — that was one ~200MB orphan per minute until the VM wedged.
    applied, state = _wire(monkeypatch, tmp_path)
    monkeypatch.setattr(approvals, "_apply_openclaw_exec_preset",
                        lambda preset: (applied.append(preset), False)[1])
    t = {"now": 1000.0}
    monkeypatch.setattr(approvals.time, "time", lambda: t["now"])
    approvals.sync_openclaw_exec_policy([RM])
    approvals.sync_openclaw_exec_policy([RM])  # inside backoff window
    assert applied == ["cautious"]  # second call did not shell out
    assert not state.exists()       # never written on failure
    t["now"] += approvals._EXEC_POLICY_BACKOFF_BASE_S + 1
    approvals.sync_openclaw_exec_policy([RM])  # backoff expired → retry
    assert applied == ["cautious", "cautious"]


def test_backoff_escalates_and_caps(monkeypatch, tmp_path):
    applied, _ = _wire(monkeypatch, tmp_path)
    monkeypatch.setattr(approvals, "_apply_openclaw_exec_preset",
                        lambda preset: (applied.append(preset), False)[1])
    t = {"now": 1000.0}
    monkeypatch.setattr(approvals.time, "time", lambda: t["now"])
    delays = []
    for _ in range(6):
        approvals.sync_openclaw_exec_policy([RM])
        delays.append(approvals._EXEC_POLICY_BACKOFF["until"] - t["now"])
        t["now"] = approvals._EXEC_POLICY_BACKOFF["until"] + 1
    assert delays == [300, 600, 1200, 2400, 3600, 3600]  # 2x, capped at 1h


def test_success_resets_backoff(monkeypatch, tmp_path):
    applied, state = _wire(monkeypatch, tmp_path)
    ok = {"v": False}
    monkeypatch.setattr(approvals, "_apply_openclaw_exec_preset",
                        lambda preset: (applied.append(preset), ok["v"])[1])
    t = {"now": 1000.0}
    monkeypatch.setattr(approvals.time, "time", lambda: t["now"])
    approvals.sync_openclaw_exec_policy([RM])   # fails → backoff armed
    assert approvals._EXEC_POLICY_BACKOFF["fails"] == 1
    ok["v"] = True
    t["now"] += approvals._EXEC_POLICY_BACKOFF_BASE_S + 1
    approvals.sync_openclaw_exec_policy([RM])   # succeeds
    assert state.read_text() == "cautious"
    assert approvals._EXEC_POLICY_BACKOFF == {"fails": 0, "until": 0.0}


def test_timeout_kills_whole_process_group(monkeypatch, tmp_path):
    # `openclaw` is a wrapper: on timeout the node CHILD must die too, not
    # just the wrapper (the orphan leak that wedged the VM, 2026-07-10).
    import os
    import time as _time
    child_pid_file = tmp_path / "child.pid"
    fake = tmp_path / "openclaw"
    fake.write_text("#!/bin/bash\nsleep 300 &\necho $! > %s\nwait\n"
                    % child_pid_file)
    fake.chmod(0o755)
    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: (str(fake), dict(os.environ)))
    monkeypatch.setattr(approvals, "_EXEC_POLICY_APPLY_TIMEOUT_S", 1)
    assert approvals._apply_openclaw_exec_preset("cautious") is False
    deadline = _time.time() + 5
    child = int(child_pid_file.read_text().strip())
    while _time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break  # grandchild is gone — group kill worked
        _time.sleep(0.1)
    else:
        os.kill(child, 9)  # cleanup before failing
        raise AssertionError("grandchild survived the process-group kill")


def test_want_detector():
    assert approvals._policies_want_exec_gate([RM]) is True
    assert approvals._policies_want_exec_gate([SECRETS]) is True
    assert approvals._policies_want_exec_gate([]) is False
    assert approvals._policies_want_exec_gate(
        [{"action": "monitor", "tool": "exec"}]) is False  # monitor != gate


def test_native_approval_poll_imports_pending_rows(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.rows = []
            self.preserve_resolved = []

        def query_approvals(self, **kwargs):
            rows = self.rows
            if kwargs.get("status"):
                rows = [row for row in rows
                        if row.get("status") == kwargs["status"]]
            return rows

        def ingest_approval(self, row, preserve_resolved=False):
            self.rows.append(row)
            self.preserve_resolved.append(preserve_resolved)

    store = FakeStore()
    import clawmetry.local_store as local_store
    monkeypatch.setattr(local_store, "get_store", lambda: store)
    monkeypatch.setattr(approvals, "_native_approval_poll_thread", None)
    monkeypatch.setattr(
        approvals, "_native_approval_poll_result",
        (True, {"approvals": [{
            "id": "ap_1", "sessionId": "sess-1", "summary": "run rm",
            "createdAtMs": 123,
        }]}, ""),
    )
    monkeypatch.setattr(approvals, "_native_approval_poll_at", 0.0)

    assert approvals.poll_openclaw_approvals() == 1
    row = store.rows[0]
    assert row["id"] == "ap_1"
    assert row["requestor_session_id"] == "openclaw:sess-1"
    assert row["args"]["source"] == "openclaw-native"
    assert "preserve_resolved" not in row["args"]
    assert store.preserve_resolved == [True]


def test_native_approval_resolve_maps_decisions(monkeypatch):
    calls = []
    monkeypatch.setattr(
        approvals, "_run_openclaw_approval_command",
        lambda args: (calls.append(args), (True, None, ""))[1],
    )

    assert approvals.resolve_openclaw_approval("ap_1", "approve") is True
    assert approvals.resolve_openclaw_approval("ap_2", "deny", "not expected") is True
    assert calls == [
        ["approvals", "resolve", "ap_1", "allow-once"],
        ["approvals", "resolve", "ap_2", "deny", "--reason", "not expected"],
    ]


class _FakeProc:
    """Minimal Popen stand-in: `communicate` either returns or times out."""

    def __init__(self, returncode=0, stdout="", stderr="", timeout=False):
        self.pid = 4242
        self.returncode = returncode
        self._out, self._err = stdout, stderr
        self._timeout = timeout
        self.killed_group = False
        self.killed_proc = False

    def communicate(self, timeout=None):
        if self._timeout:
            raise subprocess.TimeoutExpired("openclaw", timeout)
        return self._out, self._err

    def kill(self):
        self.killed_proc = True

    def wait(self, timeout=None):
        return self.returncode


def _patch_cli(monkeypatch, proc):
    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: ("/usr/local/bin/openclaw", {}))
    seen = {}

    def _popen(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)
    return seen


def test_native_cli_nonzero_exit_is_fail_open(monkeypatch):
    proc = _FakeProc(returncode=2, stderr="gateway unavailable")
    _patch_cli(monkeypatch, proc)

    ok, payload, error = approvals._run_openclaw_approval_command(
        ["approvals", "pending", "--json"])
    assert (ok, payload) == (False, None)
    assert "gateway unavailable" in error


def test_native_cli_timeout_kills_the_process_group(monkeypatch):
    """`openclaw` is a wrapper; killing only the wrapper leaves its node
    child running. A poll that runs every few seconds cannot leak one of
    those per timeout, so the CLI gets its own session and the GROUP is
    killed (same reason _apply_openclaw_exec_preset does it)."""
    proc = _FakeProc(timeout=True)
    seen = _patch_cli(monkeypatch, proc)
    killed = []
    monkeypatch.setattr(approvals.os, "killpg",
                        lambda pid, sig: killed.append((pid, sig)))

    ok, payload, error = approvals._run_openclaw_approval_command(
        ["approvals", "pending", "--json"])
    assert (ok, payload) == (False, None)
    assert error == "openclaw command timed out"
    assert seen["kwargs"].get("start_new_session") is True
    assert killed == [(proc.pid, signal.SIGKILL)]


def test_native_cli_parses_documented_payload(monkeypatch):
    """`openclaw approvals pending --json` emits {"approvals": [...]} at the
    top level (docs/cli/approvals.md)."""
    _patch_cli(monkeypatch, _FakeProc(stdout='{"approvals": [{"id": "x"}]}'))
    ok, payload, error = approvals._run_openclaw_approval_command(
        ["approvals", "pending", "--json"])
    assert (ok, error) == (True, "")
    assert payload == {"approvals": [{"id": "x"}]}


def test_native_poll_runs_cli_off_watcher_thread(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def _slow_command(args):
        started.set()
        release.wait(2)
        return True, {"approvals": []}, ""

    monkeypatch.setattr(approvals, "_run_openclaw_approval_command", _slow_command)
    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: ("/usr/local/bin/openclaw", {}))
    monkeypatch.setattr(approvals, "_native_approval_poll_at", 0.0)
    monkeypatch.setattr(approvals, "_native_approval_poll_thread", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_result", None)

    assert approvals.poll_openclaw_approvals() == 0
    assert started.wait(1)
    release.set()
    thread = approvals._native_approval_poll_thread
    assert thread is not None
    thread.join(1)


def test_native_poll_reconciles_missing_and_preserves_resolved(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.rows = [
                {"id": "resolved", "status": "approved",
                 "args": {"source": "openclaw-native"}},
                {"id": "stale", "status": "pending",
                 "args": {"source": "openclaw-native"}},
            ]
            self.ingested = []
            self.decisions = []

        def query_approvals(self, **kwargs):
            rows = self.rows
            if kwargs.get("status"):
                rows = [r for r in rows if r["status"] == kwargs["status"]]
            return rows

        def ingest_approval(self, row, preserve_resolved=False):
            self.ingested.append((row, preserve_resolved))

        def update_approval_decision(self, *args):
            self.decisions.append(args)

    store = FakeStore()
    import clawmetry.local_store as local_store
    monkeypatch.setattr(local_store, "get_store", lambda: store)
    monkeypatch.setattr(approvals, "_native_approval_poll_thread", None)
    monkeypatch.setattr(
        approvals, "_native_approval_poll_result",
        (True, {"approvals": [{"id": "resolved", "summary": "already done"},
                              {"id": "new", "summary": "new request"}]}, ""),
    )

    assert approvals.poll_openclaw_approvals() == 1
    assert [(row["id"], preserve) for row, preserve in store.ingested] == [
        ("new", True)
    ]
    assert store.decisions == [
        ("stale", "expired", "openclaw-sync", "no longer pending in OpenClaw")
    ]


def test_watcher_loop_integrates_native_poll(monkeypatch):
    stop = threading.Event()
    calls = []

    monkeypatch.setattr(approvals, "load_policies", lambda api_key=None: [])
    monkeypatch.setattr(approvals, "poll_openclaw_approvals",
                        lambda: calls.append("poll") or 0)
    monkeypatch.setattr(approvals, "sync_runtime_gates", lambda policies: None)

    def _watch(*args, **kwargs):
        calls.append("watch")
        stop.set()
        return 0

    monkeypatch.setattr(approvals, "watch_iteration", _watch)
    approvals.watcher_loop("", "node", interval_sec=0, stop_event=stop)
    assert calls == ["poll", "watch"]


def _patch_native_resolver(monkeypatch, verdict):
    """Stub resolve_openclaw_approval on the LIVE approvals module.

    routes/policy.py imports it late (`from clawmetry import approvals`), so
    it resolves whatever sits in sys.modules at request time. Several test
    files importlib.reload() this module, after which the name bound at the
    top of THIS file is a dead object and patching it lets the route shell
    out to the real `openclaw` CLI. Returns the call log."""
    import sys as _sys
    live = _sys.modules["clawmetry.approvals"]
    calls = []
    monkeypatch.setattr(
        live, "resolve_openclaw_approval",
        lambda *args, **kwargs: (calls.append((args, kwargs)), verdict)[1])
    return calls


def test_native_route_does_not_update_local_status_when_resolve_fails(monkeypatch):
    from flask import Flask
    import routes.local_query as local_query
    import routes.policy as policy

    import clawmetry.entitlements as entitlements
    entitlement = entitlements.Entitlement(
        tier="pro", source="test", grace=False,
        features=frozenset({"approval_queue"}), runtimes=frozenset(),
    )
    monkeypatch.setattr(entitlements, "get_entitlement",
                        lambda force=False: entitlement)
    monkeypatch.setattr(policy, "_ls_call", lambda *args, **kwargs: [{
        "id": "native-1", "status": "pending",
        "args": {"source": "openclaw-native"},
    }])
    updates = []
    monkeypatch.setattr(local_query, "local_store_via_daemon",
                        lambda method, **kwargs: updates.append((method, kwargs)))
    resolved = _patch_native_resolver(monkeypatch, False)

    app = Flask(__name__)
    app.register_blueprint(policy.bp_policy)
    response = app.test_client().post(
        "/api/approvals/native-1/decide", json={"decision": "approve"})
    assert response.status_code == 502
    assert updates == []
    # Without this the test passes for the wrong reason: an unpatched
    # resolver shells out to the real `openclaw` and also returns False.
    assert len(resolved) == 1


def test_native_route_updates_local_status_after_resolve(monkeypatch):
    from flask import Flask
    import routes.local_query as local_query
    import routes.policy as policy

    import clawmetry.entitlements as entitlements
    entitlement = entitlements.Entitlement(
        tier="pro", source="test", grace=False,
        features=frozenset({"approval_queue"}), runtimes=frozenset(),
    )
    monkeypatch.setattr(entitlements, "get_entitlement",
                        lambda force=False: entitlement)
    monkeypatch.setattr(policy, "_ls_call", lambda *args, **kwargs: [{
        "id": "native-2", "status": "pending",
        "args": {"source": "openclaw-native"},
    }])
    updates = []
    monkeypatch.setattr(local_query, "local_store_via_daemon",
                        lambda method, **kwargs: updates.append((method, kwargs)) or 1)
    resolved = _patch_native_resolver(monkeypatch, True)

    app = Flask(__name__)
    app.register_blueprint(policy.bp_policy)
    response = app.test_client().post(
        "/api/approvals/native-2/decide",
        json={"decision": "deny", "reason": "not expected"},
    )
    assert response.status_code == 200
    assert updates == [("update_approval_decision", {
        "approval_id": "native-2", "decision": "deny",
        "resolver": "local", "reason": "not expected",
    })]
    assert len(resolved) == 1


# ── the real `openclaw approvals pending --json` entry shape ──────────────
# Taken from readPendingApprovalEntry in the shipped CLI bundle and from
# docs/cli/approvals.md: normalized entries under "approvals", each
# {id, kind, agentId, sessionKey, createdAtMs, expiresAtMs, summary}.
# There is no sessionId and no rawId in that payload.
_REAL_ENTRY = {
    "id": "exec_01HV9ZZ",
    "kind": "exec",
    "agentId": "main",
    "sessionKey": "sess-abc123",
    "createdAtMs": 1758300000000,
    "expiresAtMs": 1758300300000,
    "summary": "rm -rf /tmp/build",
}


class _RecordingStore:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.ingested = []
        self.decisions = []

    def query_approvals(self, **kwargs):
        rows = self.rows
        if kwargs.get("status"):
            rows = [r for r in rows if r.get("status") == kwargs["status"]]
        return rows

    def ingest_approval(self, row, preserve_resolved=False):
        self.ingested.append((row, preserve_resolved))

    def update_approval_decision(self, *args):
        self.decisions.append(args)


def _arm_poll(monkeypatch, store, payload):
    import clawmetry.local_store as local_store
    monkeypatch.setattr(local_store, "get_store", lambda: store)
    monkeypatch.setattr(approvals, "_native_approval_poll_thread", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_result", payload)
    monkeypatch.setattr(approvals, "_native_approval_poll_at", 0.0)
    monkeypatch.setattr(approvals, "_NATIVE_APPROVAL_BACKOFF",
                        {"fails": 0, "until": 0.0})


def test_native_poll_reads_the_fields_openclaw_actually_emits(monkeypatch):
    """OpenClaw names the session `sessionKey`. Reading `sessionId` left
    requestor_session_id NULL, which is the column the Approvals tab scopes
    and attributes every row by."""
    store = _RecordingStore()
    _arm_poll(monkeypatch, store, (True, {"approvals": [_REAL_ENTRY]}, ""))

    assert approvals.poll_openclaw_approvals() == 1
    row, preserve = store.ingested[0]
    assert preserve is True
    assert row["requestor_session_id"] == "openclaw:sess-abc123"
    assert row["action"] == "rm -rf /tmp/build"
    assert row["args"]["agent_id"] == "main"
    assert row["args"]["expires_at_ms"] == 1758300300000
    # The runtime stopped to ask; it was not a ClawMetry policy firing.
    assert row["args"]["kind"] == "permission_prompt"
    assert row["args"]["runtime"] == "openclaw"
    # routes/policy.py::_arg_preview reads `command` — without it the card
    # previews the raw native blob instead of the command being approved.
    assert row["args"]["command"] == "rm -rf /tmp/build"
    assert row["args"]["native"] == _REAL_ENTRY


def test_native_poll_stores_created_at_as_iso(monkeypatch):
    """createdAtMs is epoch milliseconds; the approvals table holds ISO-8601
    and /api/approvals hands created_at to the UI verbatim."""
    store = _RecordingStore()
    _arm_poll(monkeypatch, store, (True, {"approvals": [_REAL_ENTRY]}, ""))
    approvals.poll_openclaw_approvals()

    created = store.ingested[0][0]["created_at"]
    assert created.endswith("Z") and created[4] == "-" and created[10] == "T"
    from datetime import datetime
    datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")  # parses, or raises


def test_native_poll_survives_a_kick_inside_the_poll_window(monkeypatch):
    """watcher_loop is kick-driven (every tool_call), so it re-enters well
    inside _NATIVE_APPROVAL_POLL_INTERVAL_S with no result waiting."""
    store = _RecordingStore()
    _arm_poll(monkeypatch, store, (True, {"approvals": []}, ""))
    assert approvals.poll_openclaw_approvals() == 0      # consumes the result
    monkeypatch.setattr(approvals, "_native_approval_poll_at", time.time())
    assert approvals.poll_openclaw_approvals() == 0      # must not raise


def test_native_poll_backs_off_a_failing_gateway(monkeypatch):
    """Gateway-down is ordinary on a box where OpenClaw is installed but not
    running. Without a backoff the kick-driven watcher spawns one node
    process every few seconds, forever."""
    spawned = []
    monkeypatch.setattr(approvals, "_native_approval_poll_thread", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_at", 0.0)
    monkeypatch.setattr(approvals, "_NATIVE_APPROVAL_BACKOFF",
                        {"fails": 0, "until": 0.0})
    monkeypatch.setattr(
        approvals, "_native_approval_poll_result",
        (False, None, "Gateway not reachable at ws://127.0.0.1:18789"))

    class _Thread:
        def __init__(self, **kwargs):
            spawned.append(kwargs.get("name"))

        def is_alive(self):
            return False

        def start(self):
            pass

    monkeypatch.setattr(approvals.threading, "Thread", _Thread)

    assert approvals.poll_openclaw_approvals() == 0
    assert approvals._NATIVE_APPROVAL_BACKOFF["until"] > time.time()
    monkeypatch.setattr(approvals, "_native_approval_poll_result", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_at", 0.0)
    assert approvals.poll_openclaw_approvals() == 0
    assert spawned == []          # backed off: no new CLI process


def test_native_poll_sweep_only_touches_pending_native_rows(monkeypatch):
    """A row decided here stays decided; only a still-pending native row
    that OpenClaw has stopped listing is expired."""
    store = _RecordingStore(rows=[
        {"id": "decided", "status": "approved",
         "args": {"source": "openclaw-native"}},
        {"id": "stale", "status": "pending",
         "args": {"source": "openclaw-native"}},
        {"id": "ours", "status": "pending",
         "args": {"source": "pretooluse-hook"}},
    ])
    _arm_poll(monkeypatch, store,
              (True, {"approvals": [{"id": "decided", "summary": "x"},
                                    {"id": "new", "summary": "y"}]}, ""))

    assert approvals.poll_openclaw_approvals() == 1
    assert [r["id"] for r, _ in store.ingested] == ["new"]
    assert store.decisions == [
        ("stale", "expired", "openclaw-sync", "no longer pending in OpenClaw")
    ]


def test_native_remember_always_maps_to_allow_always(monkeypatch):
    """"Approve & always allow" must reach OpenClaw as allow-always, or the
    remembered rule exists only on our side and OpenClaw asks again."""
    calls = []
    monkeypatch.setattr(
        approvals, "_run_openclaw_approval_command",
        lambda args: (calls.append(args), (True, None, ""))[1],
    )

    assert approvals.resolve_openclaw_approval("a1", "approve") is True
    assert approvals.resolve_openclaw_approval(
        "a2", "approve", remember="always") is True
    assert approvals.resolve_openclaw_approval(
        "a3", "approve", remember="session") is True
    assert approvals.resolve_openclaw_approval(
        "a4", "deny", "no", remember="always") is True
    assert calls == [
        ["approvals", "resolve", "a1", "allow-once"],
        ["approvals", "resolve", "a2", "allow-always"],
        ["approvals", "resolve", "a3", "allow-once"],
        ["approvals", "resolve", "a4", "deny", "--reason", "no"],
    ]


def test_native_poll_skipped_without_openclaw(monkeypatch):
    """The approvals watcher runs on every daemon. Most nodes run some other
    runtime, and there the poll must cost a `which`, not a node process."""
    spawned = []
    monkeypatch.setattr(approvals, "_native_approval_poll_thread", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_result", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_at", 0.0)
    monkeypatch.setattr(approvals, "_NATIVE_APPROVAL_BACKOFF",
                        {"fails": 0, "until": 0.0})
    monkeypatch.setattr(approvals.threading, "Thread",
                        lambda **kw: spawned.append(kw) or _NeverThread())

    monkeypatch.setattr(approvals, "_openclaw_env_and_bin", lambda: (None, {}))
    assert approvals.poll_openclaw_approvals() == 0
    assert spawned == []

    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: ("/usr/local/bin/openclaw", {}))
    assert approvals.poll_openclaw_approvals() == 0
    assert len(spawned) == 1          # present -> it does poll


def test_native_poll_env_kill_switch(monkeypatch):
    spawned = []
    monkeypatch.setattr(approvals, "_native_approval_poll_thread", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_result", None)
    monkeypatch.setattr(approvals, "_native_approval_poll_at", 0.0)
    monkeypatch.setattr(approvals, "_NATIVE_APPROVAL_BACKOFF",
                        {"fails": 0, "until": 0.0})
    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: ("/usr/local/bin/openclaw", {}))
    monkeypatch.setattr(approvals.threading, "Thread",
                        lambda **kw: spawned.append(kw) or _NeverThread())

    monkeypatch.setenv("CLAWMETRY_NATIVE_APPROVALS", "0")
    assert approvals.poll_openclaw_approvals() == 0
    assert spawned == []


class _NeverThread:
    def is_alive(self):
        return False

    def start(self):
        pass
