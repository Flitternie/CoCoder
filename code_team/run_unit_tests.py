#!/usr/bin/env python3
"""
Standalone script: copies unit_tests into a completed run and executes pytest.

Tests run inside a temporary venv to avoid polluting the host environment.
Uses TEST_PYTHON_PATH from .env to create the venv (e.g. Python 3.10).

Usage:
    python run_unit_tests.py <run_id>
        [--timeout SECONDS]           # per-test timeout (pytest-timeout + --forked)
        [--overall-timeout SECONDS]   # wall-clock ceiling for the whole run (default 1800)
        [--skip-setup]
        [--keep-venv]

Example:
    python run_unit_tests.py 20260216_011456_DevEval_TextCNN
    python run_unit_tests.py 20260216_011456_DevEval_TextCNN --timeout 120
    python run_unit_tests.py 20260216_011456_DevEval_TextCNN --timeout 10 --overall-timeout 600

Interrupt behavior:
    First  Ctrl+C → SIGTERM the child group, 3s grace, then SIGKILL.
                    Partial summary.json is still written.
    Second Ctrl+C → hard exit, no summary.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import load_repo_bundle
from common.workspace_builder import copy_tests_for_final_eval, _copy_path

VENV_DIR_NAME = ".test_venv"
ENV_FILE = PROJECT_ROOT.parent / ".env"
_ORPHAN_KILL_ATTEMPTS = 3
TEST_INFRA_PACKAGES = (
    "pytest==8.4.2",
    "pytest-timeout==2.4.0",
    "pytest-json-report==1.5.0",
    "pytest-cov==4.1.0",
    "pytest-forked==1.6.0",
)


def _load_test_python() -> str:
    """Read TEST_PYTHON_PATH from .env; error if missing or invalid."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("TEST_PYTHON_PATH=") and not line.startswith("#"):
                path = line.split("=", 1)[1].strip()
                p = Path(path)
                if p.exists():
                    return str(p)
                print(f"Error: TEST_PYTHON_PATH={path} does not exist")
                sys.exit(1)
    print(f"Error: TEST_PYTHON_PATH not set in {ENV_FILE}")
    print("Please add e.g.: TEST_PYTHON_PATH=/path/to/python3.10")
    sys.exit(1)


def _collect_descendants(root_pid: int) -> list[int]:
    """Recursively collect all descendant PIDs of `root_pid` via PPID chain."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,ppid"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.SubprocessError, OSError):
        return []
    children_map: dict[int, list[int]] = {}
    for line in out.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                pid, ppid = int(parts[0]), int(parts[1])
                children_map.setdefault(ppid, []).append(pid)
            except ValueError:
                pass
    result: list[int] = []
    stack = [root_pid]
    while stack:
        p = stack.pop()
        for child in children_map.get(p, []):
            result.append(child)
            stack.append(child)
    return result


def _kill_orphans(pids: list[int]) -> None:
    """SIGKILL a list of PIDs, retrying until none remain (up to 3 attempts)."""
    import time as _time
    my_pid = os.getpid()
    remaining = [p for p in pids if p != my_pid]
    for _ in range(_ORPHAN_KILL_ATTEMPTS):
        alive = []
        for pid in remaining:
            try:
                os.kill(pid, 0)  # check if alive
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
            alive.append(pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if not alive:
            return
        print(f"[run_unit_tests] killing {len(alive)} orphan process(es): {alive}", flush=True)
        _time.sleep(0.5)
        remaining = alive


def _build_test_cmd(bundle, workspace_dir: Path) -> str:
    """Build the pytest command string from bundle config."""
    unit_cmd = bundle.config.get("unit_test_script")
    if not unit_cmd:
        unit_tests_dir = workspace_dir / "unit_tests"
        if not unit_tests_dir.exists():
            print("Error: unit_tests directory does not exist in workspace")
            sys.exit(1)
        unit_cmd = "python -m pytest unit_tests --continue-on-collection-errors -v"
    else:
        # Rewrite bare pytest/py.test/coverage to 'python -m' form so the
        # venv's python is used (bare executables may resolve to conda base).
        parts = unit_cmd.split()
        if parts[0] in ("pytest", "py.test"):
            parts[0] = "python -m pytest"
            unit_cmd = " ".join(parts)
        elif parts[0] == "coverage":
            parts[0] = "python -m coverage"
            unit_cmd = " ".join(parts)

        # Fix test directory if it doesn't exist in workspace
        _SKIP = {"pytest", "python", "py.test", "coverage", "run", "-m"}
        parts = unit_cmd.split()
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if part.startswith("-") or "=" in part or "." in part or part in _SKIP:
                continue
            test_dir = workspace_dir / part
            if not test_dir.exists() and (workspace_dir / "unit_tests").exists():
                print(f"Warning: test directory '{part}' not found, using 'unit_tests' instead")
                parts[i] = "unit_tests"
                unit_cmd = " ".join(parts)
            break

    if "--continue-on-collection-errors" not in unit_cmd:
        unit_cmd += " --continue-on-collection-errors"
    if " -s" not in unit_cmd and "--capture=no" not in unit_cmd:
        unit_cmd += " -s"
    if "--color" not in unit_cmd:
        unit_cmd += " --color=yes"
    return unit_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute unit_tests for a completed run")
    parser.add_argument("run_id", help="Run ID, e.g. 20260216_011456_DevEval_TextCNN")
    parser.add_argument("--timeout", type=int, default=None, help="Per-test timeout in seconds")
    parser.add_argument("--overall-timeout", type=int, default=1800,
                        help="Wall-clock ceiling for the whole pytest run (seconds); "
                             "fires SIGKILL on the process group if exceeded. Default 1800.")
    parser.add_argument("--skip-setup", action="store_true", help="Skip dependency installation and setup script")
    parser.add_argument("--keep-venv", action="store_true",
                        help="Keep the temporary venv after tests finish (useful for debugging)")
    args = parser.parse_args()

    # --- Load run metadata ---
    # Recursively search results/ for the matching run_id.
    results_root = PROJECT_ROOT / "results"
    run_dir = None
    if results_root.is_dir():
        for meta_file in results_root.rglob("runs/meta.json"):
            # meta_file = results/<...>/<run_id>/runs/meta.json
            candidate = meta_file.parent  # .../runs
            run_id_dir = candidate.parent  # .../<run_id>
            if run_id_dir.name == args.run_id:
                run_dir = candidate
                break
    if not run_dir:
        print(f"Error: run_id '{args.run_id}' not found under {results_root}")
        sys.exit(1)

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = meta["run_config"]
    # Prefer workspace_dir from meta.json; fall back to sibling workspaces/.
    ws_cfg = cfg.get("workspace_dir", "")
    workspace_dir = None
    if ws_cfg:
        ws_path = Path(ws_cfg)
        resolved = ws_path.resolve() if ws_path.is_absolute() else (PROJECT_ROOT / ws_path).resolve()
        if resolved.exists():
            workspace_dir = resolved
    if not workspace_dir:
        # Fallback: run_dir = .../runs, workspace_dir = sibling workspaces/.
        workspace_dir = (run_dir.parent / "workspaces").resolve()
    if not workspace_dir.exists():
        print(f"Error: workspace does not exist: {workspace_dir}")
        sys.exit(1)

    bundle = load_repo_bundle(cfg["dataset"], cfg["repo"])

    # --- Copy unit_tests into workspace ---
    copied = copy_tests_for_final_eval(bundle, workspace_dir, "unit_tests")
    if copied:
        print(f"Copied {copied} -> {workspace_dir / copied}")
    else:
        print("unit_tests does not need to be copied (not configured or already exists)")

    # --- Copy setup_shell_script if needed ---
    if not args.skip_setup:
        setup_script = bundle.config.get("setup_shell_script")
        if setup_script:
            src = Path(bundle.repo_dir) / setup_script
            if src.exists() and not (workspace_dir / setup_script).exists():
                _copy_path(src, workspace_dir / setup_script)
                print(f"Copied {setup_script} -> {workspace_dir / setup_script}")

    # --- Build test command ---
    unit_cmd = _build_test_cmd(bundle, workspace_dir)
    if args.timeout is not None and "--timeout" not in unit_cmd:
        unit_cmd += f" --timeout={args.timeout}"
        # NOTE: We intentionally use the default signal-based timeout method.
        # The thread-based method (--timeout-method=thread) cannot actually
        # interrupt stuck code (e.g. infinite loops) — it only prints a stack
        # trace while the test keeps running, blocking the entire suite.
        # The signal method sends SIGALRM which kills the test and lets
        # pytest continue to the next one.
        # --forked runs each test in a fork; if SIGALRM is swallowed (C ext,
        # custom signal handlers), the parent still reaps the frozen child
        # and moves on.
        if "--forked" not in unit_cmd:
            unit_cmd += " --forked"

    # --- Resolve Python for venv ---
    test_python = _load_test_python()
    print(f"Test Python: {test_python}")

    # --- Build bash script: venv → install → test ---
    venv_path = workspace_dir / VENV_DIR_NAME
    if venv_path.exists():
        shutil.rmtree(venv_path)

    steps = [
        f"{test_python} -m venv {VENV_DIR_NAME}",
        f"source {VENV_DIR_NAME}/bin/activate",
        "pip install --upgrade pip -q",
    ]

    if not args.skip_setup:
        deps = bundle.config.get("dependencies")
        if deps and (workspace_dir / deps).exists():
            steps.append(f"pip install -r {deps}")
        setup_script = bundle.config.get("setup_shell_script")
        if setup_script and (workspace_dir / setup_script).exists():
            steps.append(f"sh {setup_script}")
        if (workspace_dir / "setup.py").exists() or (workspace_dir / "pyproject.toml").exists():
            steps.append("pip install -e .")

    # Always ensure test-runner infrastructure packages are available at the
    # runner's known-good versions.  This is installed AFTER project deps so
    # stale pins in requirements.txt (for example pytest-cov==4.0.0, which
    # lacks --cov-report=json support) cannot override the runner contract.
    steps.append(f"pip install --upgrade {' '.join(TEST_INFRA_PACKAGES)} -q")

    steps.append(f"PYTHONPATH=. {unit_cmd}")

    bash_script = " && \\\n  ".join(steps)

    print(f"\nRunning in: isolated venv ({venv_path})")
    print(f"Working directory: {workspace_dir}")
    print(f"\n--- bash script ---\n{bash_script}\n---\n")
    print("=" * 60)

    # Stream output in real-time while capturing for later parsing.
    # NOTE: We read character-by-character instead of line-by-line because
    # pytest writes progress dots (e.g. "...FF..F") for each test file on a
    # SINGLE LINE without newlines until the file finishes.  With line-buffered
    # reading (`for line in proc.stdout`), the reader blocks until the entire
    # line is complete – which can take hours for files with many slow tests.
    proc = subprocess.Popen(
        bash_script,
        shell=True,
        cwd=workspace_dir,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,       # binary mode for reliable single-byte reads
        start_new_session=True,  # put child in its own process group
    )

    # On Ctrl+C, nuke the entire process group so bash + pytest + any
    # grandchildren die together instead of becoming orphans — but do NOT
    # sys.exit here; let the main loop fall through to summary parsing so
    # partial results still get persisted.
    interrupted = {"signum": None}

    def _kill_group(signum, _):
        if interrupted["signum"] is None:
            interrupted["signum"] = signum
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        # Second signal → user is impatient; escalate and hard-exit.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os._exit(130 if signum == signal.SIGINT else 143)

    prev_sigint = signal.signal(signal.SIGINT, _kill_group)
    prev_sigterm = signal.signal(signal.SIGTERM, _kill_group)

    # Read in non-blocking chunks so we can enforce a wall-clock ceiling.
    # Single-byte reads are preserved for pytest's progress-dot output,
    # but we use select() so a stuck child never hangs us indefinitely.
    import select
    import time as _time

    assert proc.stdout is not None
    os.set_blocking(proc.stdout.fileno(), False)
    captured_chunks: list[bytes] = []
    captured_size = 0
    MAX_CAPTURE_BYTES = 2 * 1024 * 1024  # 2MB cap; only tail is needed for parsing
    timed_out = False
    start_ts = _time.monotonic()
    deadline = start_ts + args.overall_timeout if args.overall_timeout else None
    INTERRUPT_GRACE_SEC = 3.0
    interrupt_kill_ts: float | None = None
    # Periodically snapshot all descendant PIDs so we can kill orphans that
    # escaped the process group (pytest-forked calls setpgrp). Once a child's
    # parent dies, it gets reparented to PID 1 and _collect_descendants can no
    # longer find it — so we accumulate throughout the run.
    all_known_descendants: set[int] = set()
    _DESCENDANT_SCAN_INTERVAL = 5.0  # seconds between scans
    _last_descendant_scan = start_ts
    try:
        while True:
            now = _time.monotonic()
            # Periodic descendant scan
            if now - _last_descendant_scan >= _DESCENDANT_SCAN_INTERVAL:
                all_known_descendants.update(_collect_descendants(proc.pid))
                _last_descendant_scan = now
            if deadline is not None and now > deadline:
                timed_out = True
                print(
                    f"\n[run_unit_tests] overall-timeout ({args.overall_timeout}s) "
                    f"exceeded; killing process group.",
                    flush=True,
                )
                break
            # Ctrl+C / SIGTERM received — start a grace-period timer, then
            # SIGKILL the group and break out so summary parsing still runs
            # even if the child is wedged (e.g. pytest stuck during shutdown
            # after printing the summary line).
            if interrupted["signum"] is not None:
                if interrupt_kill_ts is None:
                    interrupt_kill_ts = now + INTERRUPT_GRACE_SEC
                    print(
                        f"\n[run_unit_tests] interrupted; waiting up to "
                        f"{INTERRUPT_GRACE_SEC:.0f}s for graceful exit.",
                        flush=True,
                    )
                elif now > interrupt_kill_ts:
                    print(
                        "\n[run_unit_tests] grace period exceeded; SIGKILL.",
                        flush=True,
                    )
                    all_known_descendants.update(_collect_descendants(proc.pid))
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if all_known_descendants:
                        _kill_orphans(list(all_known_descendants))
                    break
            # Wait up to 1s for data or child exit.
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            except InterruptedError:
                ready = []
            if ready:
                chunk = proc.stdout.read(4096)
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    captured_chunks.append(chunk)
                    captured_size += len(chunk)
                    if captured_size > MAX_CAPTURE_BYTES:
                        # Evict old chunks, keep only the tail for summary parsing
                        half = MAX_CAPTURE_BYTES // 2
                        kept: list[bytes] = []
                        kept_size = 0
                        for c in reversed(captured_chunks):
                            if kept_size + len(c) > half:
                                break
                            kept.append(c)
                            kept_size += len(c)
                        kept.reverse()
                        captured_chunks = kept
                        captured_size = kept_size
                    continue
                # EOF on stdout
                break
            # No output this second; check if child already exited.
            if proc.poll() is not None:
                # Drain any final bytes still in the pipe.
                tail = proc.stdout.read()
                if tail:
                    sys.stdout.buffer.write(tail)
                    sys.stdout.buffer.flush()
                    captured_chunks.append(tail)
                break
        if timed_out:
            all_known_descendants.update(_collect_descendants(proc.pid))
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if all_known_descendants:
                _kill_orphans(list(all_known_descendants))
        else:
            # Final scan before wait — proc is still alive so PPID chain intact.
            all_known_descendants.update(_collect_descendants(proc.pid))
            proc.wait()
            if all_known_descendants:
                _kill_orphans(list(all_known_descendants))
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
    combined_bytes = b"".join(captured_chunks)
    captured_text = combined_bytes.decode("utf-8", errors="replace")

    # Build a result-like object for downstream parsing
    class _Result:
        def __init__(self, rc: int, out: str):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""
    result = _Result(proc.returncode, captured_text)

    # --- Parse pytest results from output ---
    import re as _re
    combined = (result.stdout or "") + (result.stderr or "")
    # Strip ANSI escape codes for reliable parsing and clean storage
    stripped = _re.sub(r"\x1b\[[0-9;]*m", "", combined)
    test_info: dict = {"exit_code": result.returncode}
    if interrupted["signum"] is not None:
        test_info["interrupted"] = "SIGINT" if interrupted["signum"] == signal.SIGINT else "SIGTERM"
    if timed_out:
        test_info["overall_timed_out"] = args.overall_timeout

    # Parse "collected N items" (actual pytest collection count)
    coll_m = _re.search(r"collected\s+(\d+)\s+items?", stripped)
    if coll_m:
        test_info["collected"] = int(coll_m.group(1))

    # Match pytest summary: "= 5 passed, 2 failed, 1 error in 3.21s ="
    m = _re.search(r"=+\s+([\d\w\s,]+?)\s+in\s+[\d.]+s?(?:\s*\([^)]*\))?\s*=+", stripped)
    if m:
        summary_parts = m.group(1).split(",")
        for part in summary_parts:
            part = part.strip()
            num_match = _re.match(r"(\d+)\s+(\w+)", part)
            if num_match:
                test_info[num_match.group(2)] = int(num_match.group(1))

    # --- Compute accurate total ---
    # `collected` (from "collected N items") already handles parametrize
    # expansion correctly.  `source_count` (scanning `def test_`) covers
    # files that failed to collect.  We take the max of both.
    _test_func_re = _re.compile(r"^\s*(?:async\s+)?def\s+test_", _re.MULTILINE)
    _test_dirs = [workspace_dir / d for d in ("unit_tests", "tests", "test")]
    source_count = 0
    for td in _test_dirs:
        if td.is_dir():
            for pyfile in sorted(td.rglob("test_*.py")):
                try:
                    source_count += len(_test_func_re.findall(pyfile.read_text(encoding="utf-8", errors="replace")))
                except OSError:
                    pass

    collected = test_info.get("collected", 0)
    test_info["total"] = max(collected, source_count) if source_count > 0 else collected
    if source_count > collected > 0:
        test_info["uncollected"] = source_count - collected

    # Truncate output for storage (keep last 5000 chars, ANSI-stripped)
    if len(stripped) > 5000:
        test_info["output"] = "...(truncated)...\n" + stripped[-5000:]
    else:
        test_info["output"] = stripped

    # --- Write results to summary.json ---
    summary_path = run_dir / "summary.json"
    try:
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {}
        summary["unit_test_results"] = test_info
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nTest results written to {summary_path}")
    except Exception as e:
        print(f"Warning: could not update summary.json: {e}")

    # --- Cleanup ---
    if venv_path.exists():
        if args.keep_venv:
            print(f"Keeping venv at {venv_path} (--keep-venv)")
        else:
            print(f"Cleaning up venv at {venv_path}")
            shutil.rmtree(venv_path, ignore_errors=True)

    print("=" * 60)
    print(f"\nExit code: {result.returncode}")
    if interrupted["signum"] is not None:
        sys.exit(130 if interrupted["signum"] == signal.SIGINT else 143)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
