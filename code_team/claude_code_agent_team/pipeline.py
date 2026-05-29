"""ClaudeCodePipeline: workspace setup → claude agent team run → result logging."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from common.datasets import load_repo_bundle, get_numbered_repo_name
from common.infra.artifact_logger import ArtifactLogger
from common.models import RepoBundle
from common.workspace_builder import build_workspace, copy_tests_for_final_eval

from claude_code_agent_team.config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    IterationConfig,
    RunConfig,
    RuntimeConfig,
    get_claude_model,
)
from common.config import resolve_gt_rib
from claude_code_agent_team.runtime import ClaudeCodeRuntime


class ClaudeCodePipeline:
    """Pipeline that delegates code generation to claude CLI agent teams."""

    def __init__(self, run_config: RunConfig):
        self.cfg = run_config
        numbered_repo = get_numbered_repo_name(run_config.dataset, run_config.repo)
        self.run_dir = (
            PROJECT_ROOT / "results" / run_config.dataset / numbered_repo / run_config.run_id / "runs"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_logger = ArtifactLogger(self.run_dir)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_run_config(
        dataset: str,
        repo: str,
        model: str | None = None,
        gt_rib: str | bool = False,
    ) -> RunConfig:
        resolved_model = get_claude_model(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved_model).strip("._-") or "model"
        dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._-") or "dataset"
        repo_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo).strip("._-") or "repo"
        run_id = f"{timestamp}_{model_slug}_claudecode_{dataset_slug}_{repo_slug}"

        numbered_repo = get_numbered_repo_name(dataset, repo)
        result_base = PROJECT_ROOT / "results" / dataset / numbered_repo / run_id
        run_dir = result_base / "runs"
        workspace_dir = result_base / "workspaces"
        run_dir.mkdir(parents=True, exist_ok=True)

        return RunConfig(
            run_id=run_id,
            dataset=dataset,
            repo=repo,
            run_dir=run_dir,
            workspace_dir=workspace_dir,
            iteration=IterationConfig(),
            # RuntimeConfig.model_name stores the model alias for display/logging
            runtime=RuntimeConfig(model_name=resolved_model),
            gt_rib=gt_rib,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        start = time.time()
        bundle = load_repo_bundle(self.cfg.dataset, self.cfg.repo)

        # Build workspace (copies docs, check_tests, pre-existing files)
        workspace_dir = self.cfg.workspace_dir
        manifest = build_workspace(bundle, workspace_dir)
        self.artifact_logger.write_json("artifacts/workspace_manifest.json", manifest.model_dump())

        if bundle.config.get("check_tests"):
            copy_tests_for_final_eval(bundle, workspace_dir, "check_tests")

        # Save run metadata (same format as codebase pipeline)
        meta = {
            "run_config": {
                "dataset": self.cfg.dataset,
                "repo": self.cfg.repo,
                "run_id": self.cfg.run_id,
                "run_dir": str(self.cfg.run_dir),
                "workspace_dir": str(self.cfg.workspace_dir),
                "iteration": asdict(self.cfg.iteration),
                "runtime": {"model_name": self.cfg.runtime.model_name},
            },
            "pipeline": "claude_code_agent_team",
            "model": self.cfg.runtime.model_name,
        }
        self.artifact_logger.write_json("meta.json", meta)

        # Run optional setup script
        setup_script = bundle.config.get("setup_shell_script", "")
        print(f"setup script: {setup_script}")
        if setup_script:
            script_path = workspace_dir / setup_script
            print(f"Checking for setup script at: {script_path}")
            if script_path.exists():
                print(f"[claude_code_agent_team] Running setup script: {setup_script}")
                subprocess.run(
                    ["bash", str(script_path)],
                    cwd=str(workspace_dir),
                    timeout=300,
                    capture_output=True,
                )

        # Load RIB dependency spec if configured
        rib_content = None
        rib_path = resolve_gt_rib(self.cfg, default_filename="rib_dep_ground_true.json")
        if rib_path is not None:
            rib_content = rib_path.read_text(encoding="utf-8")

        # Build initial prompt
        prompt = self._build_initial_prompt(bundle, workspace_dir, rib_content=rib_content)
        self.artifact_logger.write_text("artifacts/initial_prompt.txt", prompt)

        # Load implementation integrity doc if present for this repo
        integrity_path = DATASET_ROOT / self.cfg.dataset / self.cfg.repo / "docs" / "implementation_integrity.md"
        integrity_content = integrity_path.read_text(encoding="utf-8") if integrity_path.exists() else None

        # Run claude CLI
        runtime = ClaudeCodeRuntime(
            workspace_dir=workspace_dir,
            run_dir=self.run_dir,
            model=self.cfg.runtime.model_name,
            integrity_content=integrity_content,
        )
        pre_existing_py = {p.resolve() for p in workspace_dir.rglob("*.py")}

        result = runtime.run(prompt)
        self.artifact_logger.write_text("artifacts/agent_final_response.txt", result)

        self._collect_subagent_logs(workspace_dir)

        elapsed = time.time() - start
        agent_team_stats = self._parse_agent_team_stats()
        token_usage = self._parse_token_usage()
        delivery = self._detect_delivery(workspace_dir, pre_existing_py)
        self._write_summary(elapsed, agent_team_stats, token_usage, delivery)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_initial_prompt(self, bundle: RepoBundle, workspace_dir: Path, rib_content: str | None = None) -> str:
        cfg = bundle.config
        parts = [
            "# Project Generation Task\n",
            f"Dataset: {bundle.dataset}, Repository: {bundle.repo}\n",
            "## Requirement Documents\n",
            "The following documents are available in the workspace:\n",
        ]

        prd_path = cfg.get("PRD", "")
        uml_class_path = cfg.get("UML_class", "")
        uml_sequence_path = cfg.get("UML_sequence", "")
        arch_design_path = cfg.get("architecture_design", "")

        if not uml_class_path:
            uml_list = cfg.get("UML", [])
            if isinstance(uml_list, list) and uml_list:
                chosen = next((u for u in uml_list if "pyreverse" in u), uml_list[0])
                uml_class_path = chosen

        if prd_path:
            parts.append(f"- PRD: `{prd_path}`")
        if uml_class_path:
            parts.append(f"- UML Class Diagram: `{uml_class_path}`")
        if uml_sequence_path:
            parts.append(f"- UML Sequence Diagram: `{uml_sequence_path}`")
        if arch_design_path:
            parts.append(f"- Architecture Design: `{arch_design_path}`")

        parts.append("")
        parts.append("## Testing\n")
        check_tests = cfg.get("check_tests")
        if check_tests:
            test_cmd = cfg.get("check_test_script", "python -m pytest check_tests -v -s")
            parts.append(f"- Test directory: `{check_tests}/` (READ-ONLY, do not modify)")
            parts.append(f"- Test command: `{test_cmd}`")
        else:
            parts.append("- No tests configured. Generate code and verify syntax with py_compile.")

        if rib_content:
            parts.append("")
            parts.append("## Sub-task Specifications\n")
            parts.append("The following JSON describes the expected file structure, classes, functions, and dependencies for this project:\n")
            parts.append(f"```json\n{rib_content}\n```")

        parts.append("")
        parts.append("## Goal\n")
        parts.append("Generate a complete, working Python project that passes all tests.")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _parse_agent_team_stats(self) -> dict[str, Any]:
        """Parse logs/stream.jsonl and extract agent team spawn statistics.

        Always return a dictionary. If no Agent tool_use calls are found, return an
        "agentless" summary that includes assistant tool counts and a short
        workspace manifest summary to explain how the run completed work.
        """
        stream_log = self.run_dir / "logs" / "stream.jsonl"
        assistant_tool_calls: dict[str, int] = {}
        assistant_messages: list[str] = []

        if not stream_log.exists():
            # No stream log available; return an explicit empty summary
            return {
                "total_teams": 0,
                "total_subagents": 0,
                "agentless_run": True,
                "assistant_tool_calls": {},
                "workspace_manifest_summary": {"exists": False},
            }

        teams: list[list[dict]] = []

        for line in stream_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") != "assistant":
                continue

            # capture a short text snippet for recent assistant messages
            message_snippet = None
            content = event.get("message", {}).get("content", [])
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        txt = block.get("text", "").strip()
                        if txt:
                            texts.append(txt)
                if texts:
                    message_snippet = " ".join(texts)[:400]

            # Count all assistant tool_use calls (by tool name)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "unknown")
                        assistant_tool_calls[name] = assistant_tool_calls.get(name, 0) + 1

            # Detect Agent tool_use blocks (these indicate spawns)
            agent_calls = [
                block for block in content
                if isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == "Agent"
            ]

            if agent_calls:
                team = []
                for call in agent_calls:
                    inp = call.get("input", {})
                    team.append({
                        "subagent_type": inp.get("subagent_type", "unknown"),
                        "description": inp.get("description", ""),
                        "run_in_background": bool(inp.get("run_in_background", False)),
                    })
                teams.append(team)

            # keep last few assistant messages for diagnostics
            if message_snippet:
                assistant_messages.append(message_snippet)
                if len(assistant_messages) > 5:
                    assistant_messages.pop(0)

        # If we found at least one team, return the full per-team breakdown
        if teams:
            all_agents = [agent for team in teams for agent in team]
            total_subagents = len(all_agents)
            total_teams = len(teams)
            replacement_agents = sum(len(t) for t in teams[1:])
            parallel_spawns = sum(1 for a in all_agents if a["run_in_background"])
            sequential_spawns = total_subagents - parallel_spawns

            by_type: dict[str, int] = {}
            for agent in all_agents:
                t = agent["subagent_type"]
                by_type[t] = by_type.get(t, 0) + 1

            teams_data = []
            for i, team in enumerate(teams):
                team_by_type: dict[str, int] = {}
                for agent in team:
                    t = agent["subagent_type"]
                    team_by_type[t] = team_by_type.get(t, 0) + 1
                teams_data.append({
                    "team_index": i,
                    "phase": "initial" if i == 0 else "replacement",
                    "size": len(team),
                    "parallel_spawns": sum(1 for a in team if a["run_in_background"]),
                    "sequential_spawns": sum(1 for a in team if not a["run_in_background"]),
                    "by_type": team_by_type,
                })

            return {
                "total_teams": total_teams,
                "total_subagents": total_subagents,
                "replacement_agents": replacement_agents,
                "parallel_spawns": parallel_spawns,
                "sequential_spawns": sequential_spawns,
                "by_type": by_type,
                "teams": teams_data,
            }

        # No Agent tool_use found — produce an explicit agentless summary
        # Collect workspace manifest info if available
        manifest_path = self.run_dir / "artifacts" / "workspace_manifest.json"
        workspace_summary = {"exists": False}
        try:
            if manifest_path.exists():
                workspace_summary["exists"] = True
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict):
                    # Try common keys, else fall back to counting entries
                    file_count = None
                    if "files" in manifest and isinstance(manifest["files"], list):
                        file_count = len(manifest["files"])
                        sample = manifest["files"][:5]
                    elif "paths" in manifest and isinstance(manifest["paths"], list):
                        file_count = len(manifest["paths"])
                        sample = manifest["paths"][:5]
                    else:
                        # fallback: count top-level keys
                        file_count = len(manifest)
                        sample = list(manifest.keys())[:5]
                    workspace_summary.update({"file_count": file_count, "sample": sample})
            else:
                # Fall back to counting files in workspace directory if available
                workspace_dir = Path(self.cfg.workspace_dir) if getattr(self.cfg, "workspace_dir", None) else None
                if workspace_dir and workspace_dir.exists():
                    files = [p for p in workspace_dir.rglob("*") if p.is_file()]
                    workspace_summary.update({"exists": True, "file_count": len(files), "sample": [str(p) for p in files[:5]]})
        except Exception as e:
            print(f"[claude_code_agent_team] Failed to parse workspace manifest: {e}")

        # Also detect whether agent templates were installed into the workspace
        agents_installed = False
        try:
            workspace_dir = Path(self.cfg.workspace_dir) if getattr(self.cfg, "workspace_dir", None) else None
            if workspace_dir:
                agents_dir = workspace_dir / ".claude" / "agents"
                agents_installed = agents_dir.exists()
        except Exception as e:
            print(f"[claude_code_agent_team] Failed to check agent templates: {e}")

        return {
            "total_teams": 0,
            "total_subagents": 0,
            "agentless_run": True,
            "assistant_tool_calls": assistant_tool_calls,
            "assistant_messages_last": assistant_messages,
            "workspace_manifest_summary": workspace_summary,
            "agents_installed_in_workspace": agents_installed,
        }

    def _collect_subagent_logs(self, workspace_dir: Path) -> None:
        """Copy subagent JSONL files from ~/.claude/projects/ into logs/subagents/."""
        stream_log = self.run_dir / "logs" / "stream.jsonl"
        if not stream_log.exists():
            return

        session_id = None
        for line in stream_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                session_id = event.get("session_id")
                break

        if not session_id:
            return

        # Claude encodes the workspace path by replacing / and _ with -
        encoded = workspace_dir.resolve().as_posix().replace("/", "-").replace("_", "-")
        subagents_src = Path.home() / ".claude" / "projects" / encoded / session_id / "subagents"

        if not subagents_src.exists():
            return

        subagents_dst = self.run_dir / "logs" / "subagents"
        subagents_dst.mkdir(exist_ok=True)
        for f in subagents_src.iterdir():
            if f.name.startswith("agent-"):
                shutil.copy2(f, subagents_dst / f.name)

    def _parse_token_usage(self) -> dict[str, Any]:
        """Extract total token usage across orchestrator and all subagents."""
        stream_log = self.run_dir / "logs" / "stream.jsonl"
        if not stream_log.exists():
            return {}

        # Use the FIRST result event (orchestrator); later ones may be subagent sessions
        orch_usage: dict = {}
        litellm_cost = 0.0
        for line in stream_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                usage = event.get("usage", {})
                if usage:
                    orch_usage = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                    }
                    litellm_cost = event.get("total_cost_usd", 0) or 0.0
                    break

        if not orch_usage:
            return {}

        # Aggregate subagent usage; deduplicate turns by message ID
        sub_input = sub_output = sub_cache_read = sub_cache_creation = 0
        subagents_dir = self.run_dir / "logs" / "subagents"
        if subagents_dir.exists():
            for agent_file in sorted(subagents_dir.glob("agent-*.jsonl")):
                seen_ids: set[str] = set()
                for line in agent_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = entry.get("message")
                    if not isinstance(msg, dict):
                        continue
                    msg_id = msg.get("id")
                    usage = msg.get("usage")
                    if not msg_id or not isinstance(usage, dict) or msg_id in seen_ids:
                        continue
                    seen_ids.add(msg_id)
                    sub_input += usage.get("input_tokens", 0)
                    sub_output += usage.get("output_tokens", 0)
                    sub_cache_read += usage.get("cache_read_input_tokens", 0)
                    sub_cache_creation += usage.get("cache_creation_input_tokens", 0)

        total_input = orch_usage["input_tokens"] + sub_input
        total_output = orch_usage["output_tokens"] + sub_output
        total_cache_read = orch_usage["cache_read_input_tokens"] + sub_cache_read
        total_cache_creation = orch_usage["cache_creation_input_tokens"] + sub_cache_creation

        manual_cost = self._compute_cost(
            total_input, total_output, total_cache_read, total_cache_creation,
        )

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_input_tokens": total_cache_read,
            "cache_creation_input_tokens": total_cache_creation,
            "subagent_input_tokens": sub_input,
            "subagent_output_tokens": sub_output,
            "manual_cost_usd": round(manual_cost, 6),
            "litellm_cost_usd": litellm_cost,
        }

    def _compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
    ) -> float:
        """Compute cost from litellm's model cost map; fall back to 0 if unknown."""
        try:
            import litellm
            info = litellm.get_model_info(self.cfg.runtime.model_name)
            input_rate = info.get("input_cost_per_token", 0) or 0
            output_rate = info.get("output_cost_per_token", 0) or 0
            cache_read_rate = info.get("cache_read_input_token_cost", 0) or 0
            cache_creation_rate = info.get("cache_creation_input_token_cost", 0) or 0
            return (
                input_tokens * input_rate
                + output_tokens * output_rate
                + cache_read_tokens * cache_read_rate
                + cache_creation_tokens * cache_creation_rate
            )
        except Exception:
            return 0.0

    def _detect_delivery(self, workspace_dir: Path, pre_existing_py: set[Path]) -> dict[str, Any]:
        """Detect whether the model actually wrote any new Python files to the workspace."""
        all_py = {p.resolve() for p in workspace_dir.rglob("*.py")}
        new_files = sorted(
            str(p.relative_to(workspace_dir)) for p in (all_py - pre_existing_py)
        )
        return {
            "delivery_failure": len(new_files) == 0,
            "new_python_files": len(new_files),
            "files": new_files,
        }

    def _write_summary(
        self,
        elapsed: float,
        agent_team_stats: dict[str, Any] | None = None,
        token_usage: dict[str, Any] | None = None,
        delivery: dict[str, Any] | None = None,
    ) -> None:
        summary = {
            "run_id": self.cfg.run_id,
            "dataset": self.cfg.dataset,
            "repo": self.cfg.repo,
            "model": self.cfg.runtime.model_name,
            "pipeline": "claude_code_agent_team",
            "duration_sec": round(elapsed, 1),
        }
        if delivery:
            summary["delivery"] = delivery
        if agent_team_stats:
            summary["agent_team_stats"] = agent_team_stats
        if token_usage:
            summary["token_usage"] = token_usage
        self.artifact_logger.write_text(
            "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
