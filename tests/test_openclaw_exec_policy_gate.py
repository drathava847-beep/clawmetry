"""Daemon drives OpenClaw's native exec-approval gate from the active policies.

ClawMetry's own watcher is reactive (can't prevent a command). When a
require-approval policy covering exec is active, the daemon applies
`openclaw exec-policy preset cautious` (pre-execution gate); when none are,
it restores `yolo` — but only if it was the one that set cautious, so a
hand-set posture is never clobbered. No-op off an OpenClaw host.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

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


def test_native_cli_nonzero_exit_is_fail_open(monkeypatch):
    class Failed:
        returncode = 2
        stdout = ""
        stderr = "gateway unavailable"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Failed())
    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: ("/usr/local/bin/openclaw", {}))

    ok, payload, error = approvals._run_openclaw_approval_command(
        ["approvals", "pending", "--json"])
    assert (ok, payload) == (False, None)
    assert "gateway unavailable" in error


def test_native_cli_timeout_is_fail_open(monkeypatch):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", _timeout)
    monkeypatch.setattr(approvals, "_openclaw_env_and_bin",
                        lambda: ("/usr/local/bin/openclaw", {}))

    ok, payload, error = approvals._run_openclaw_approval_command(
        ["approvals", "pending", "--json"])
    assert (ok, payload) == (False, None)
    assert error == "openclaw command timed out"


def test_native_poll_runs_cli_off_watcher_thread(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def _slow_command(args):
        started.set()
        release.wait(2)
        return True, {"approvals": []}, ""

    monkeypatch.setattr(approvals, "_run_openclaw_approval_command", _slow_command)
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
    monkeypatch.setattr(approvals, "resolve_openclaw_approval",
                        lambda *args, **kwargs: False)

    app = Flask(__name__)
    app.register_blueprint(policy.bp_policy)
    response = app.test_client().post(
        "/api/approvals/native-1/decide", json={"decision": "approve"})
    assert response.status_code == 502
    assert updates == []


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
    monkeypatch.setattr(approvals, "resolve_openclaw_approval",
                        lambda *args, **kwargs: True)

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
