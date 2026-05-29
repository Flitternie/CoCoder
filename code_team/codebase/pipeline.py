"""Autonomous pipeline: load data → build workspace → single agent run.

Much simpler than serial/pipeline.py — the agent decides the workflow.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import time

from common.infra.artifact_logger import ArtifactLogger
from common.datasets import load_repo_bundle, get_numbered_repo_name
from common.models import RepoBundle
from common.workspace_builder import build_workspace, copy_tests_for_final_eval

from codebase.config import (
    ENABLE_PROGRESS_TRACKING,
    IterationConfig,
    PROJECT_ROOT,
    RunConfig,
    RuntimeConfig,
    WORKSPACE_ROOT,
)
from codebase.runtime import CodebaseRuntime


class CodebasePipeline:
    """Single-agent codebase pipeline."""

    def __init__(self, run_config: RunConfig):
        self.cfg = run_config
        numbered_repo = get_numbered_repo_name(run_config.dataset, run_config.repo)
        self.run_dir = PROJECT_ROOT / "results" / run_config.dataset / numbered_repo / run_config.run_id / "runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_logger = ArtifactLogger(self.run_dir)

    @staticmethod
    def create_run_config(
        dataset: str,
        repo: str,
        model_name: str | None = None,
        gt_rib: str | bool = False,
        rib_dep_tool: bool = False,
        no_visualizer: bool = False,
    ) -> RunConfig:
        from codebase.config import get_model_name, get_api_key, get_base_url
        import re

        model = get_model_name(model_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract short model name: "openai/gpt-5-mini" → "gpt-5-mini"
        model_slug = model.rsplit("/", 1)[-1]
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_slug).strip("._-") or "model"
        dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._-") or "dataset"
        repo_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo).strip("._-") or "repo"
        if gt_rib:
            rib_tag = "ribfile"
        elif rib_dep_tool:
            from common.config import get_rib_dep_model_slug
            dep_slug = get_rib_dep_model_slug() or model_slug
            rib_tag = f"ribgensimdep_{dep_slug}"
        else:
            rib_tag = "ribgensim"
        run_id = f"{timestamp}_{model_slug}_codebase_{rib_tag}_{dataset_slug}_{repo_slug}"
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
            runtime=RuntimeConfig(
                model_name=model,
                api_key=get_api_key(model),
                base_url=get_base_url(),
            ),
            iteration=IterationConfig(),  # defaults: arch=3, skeleton=3, code=10
            gt_rib=gt_rib,
            rib_dep_tool=rib_dep_tool,
            no_visualizer=no_visualizer,
        )

    def run(self) -> None:
        """Execute the codebase pipeline."""
        start = time.time()
        bundle = load_repo_bundle(self.cfg.dataset, self.cfg.repo)

        # Build workspace
        workspace_dir = self.cfg.workspace_dir
        manifest = build_workspace(bundle, workspace_dir)
        self.artifact_logger.write_json("artifacts/workspace_manifest.json", manifest.model_dump())

        # Copy check_tests if configured
        check_tests = bundle.config.get("check_tests")
        if check_tests:
            copy_tests_for_final_eval(bundle, workspace_dir, "check_tests")

        # Save run metadata (same format as serial/parallel)
        from dataclasses import asdict
        meta = {
            "run_config": {
                "dataset": self.cfg.dataset,
                "repo": self.cfg.repo,
                "run_id": self.cfg.run_id,
                "run_dir": str(self.cfg.run_dir),
                "workspace_dir": str(self.cfg.workspace_dir),
                "iteration": asdict(self.cfg.iteration),
                "runtime": {k: v for k, v in asdict(self.cfg.runtime).items() if k != "api_key"},
            },
            "pipeline": "codebase",
            "model": self.cfg.runtime.model_name,
        }
        self.artifact_logger.write_json("meta.json", meta)

        # Create runtime
        runtime = CodebaseRuntime(
            workspace_dir=workspace_dir,
            run_dir=self.run_dir,
            runtime_cfg=self.cfg.runtime,
            testfix_iters=self.cfg.iteration.testfix_iters,
            rib_dep_tool=self.cfg.rib_dep_tool,
            no_visualizer=self.cfg.no_visualizer,
        )

        # Optional progress tracker for debugging
        tracker = None
        if ENABLE_PROGRESS_TRACKING:
            from dashboard.baseline.progress_tracker import ProgressTracker
            tracker = ProgressTracker(self.run_dir, self.cfg.run_id)
            tracker.start()

        tracker_stopped = False
        try:
            # Run setup script if configured (install dependencies before agent starts)
            setup_script = bundle.config.get("setup_shell_script", "")
            if setup_script:
                import subprocess
                script_path = workspace_dir / setup_script
                if script_path.exists():
                    print(f"[codebase] Running setup script: {setup_script}")
                    subprocess.run(
                        ["bash", str(script_path)],
                        cwd=str(workspace_dir),
                        timeout=300,
                        capture_output=True,
                    )

            # If ground-truth RIB requested, copy into workspace
            from common.config import copy_gt_rib_to_workspace, resolve_gt_rib
            rib_provided = copy_gt_rib_to_workspace(self.cfg, workspace_dir)
            if rib_provided:
                print(f"[codebase] Using ground-truth RIB: {resolve_gt_rib(self.cfg)}")

            # Build the initial user prompt
            prompt = self._build_initial_prompt(bundle, workspace_dir, rib_provided=rib_provided)
            self.artifact_logger.write_text("artifacts/initial_prompt.txt", prompt)

            # Run the codebase agent
            result = runtime.run(prompt)
            self.artifact_logger.write_text("artifacts/agent_final_response.txt", result)

            # Flush tracker before building summary so the last run_tests/finish
            # events are included in summary.json.
            if tracker:
                tracker.stop()
                tracker_stopped = True

            # Build and save summary
            elapsed = time.time() - start
            self._build_summary(runtime, elapsed, tracker)

        finally:
            if tracker and not tracker_stopped:
                tracker.stop()
            runtime.close()

    def _build_initial_prompt(self, bundle: RepoBundle, workspace_dir: Path, *, rib_provided: bool = False) -> str:
        """Build the initial prompt for the codebase agent.

        Tells the agent where to find requirement docs and what tests to run.
        The agent reads the files itself using file_editor.
        """
        cfg = bundle.config
        parts = [
            "# Project Generation Task\n",
            f"Dataset: {bundle.dataset}, Repository: {bundle.repo}\n",
            "## Requirement Documents\n",
            "Read these files to understand the project:\n",
        ]

        # Document paths
        prd_path = cfg.get("PRD", "")
        uml_class_path = cfg.get("UML_class", "")
        uml_sequence_path = cfg.get("UML_sequence", "")
        arch_design_path = cfg.get("architecture_design", "")

        # Handle CodeProjectEval UML list format: pick pyreverse if available
        if not uml_class_path:
            uml_list = cfg.get("UML", [])
            if isinstance(uml_list, list) and uml_list:
                chosen = None
                for item in uml_list:
                    if "pyreverse" in item:
                        chosen = item
                        break
                uml_class_path = chosen or uml_list[0]

        if prd_path:
            parts.append(f"- PRD: `{prd_path}`")
        if uml_class_path:
            parts.append(f"- UML Class Diagram: `{uml_class_path}`")
        if uml_sequence_path:
            parts.append(f"- UML Sequence Diagram: `{uml_sequence_path}`")
        if arch_design_path:
            parts.append(f"- Architecture Design: `{arch_design_path}`")

        parts.append("")

        # Test configuration
        parts.append("## Testing\n")
        check_tests = cfg.get("check_tests")
        if check_tests:
            test_cmd = cfg.get("check_test_script", "python -m pytest check_tests -v -s")
            setup_script = cfg.get("setup_shell_script", "")
            parts.append(f"- Test directory: `{check_tests}/` (READ-ONLY, do not modify)")
            parts.append(f"- Test command: `{test_cmd}`")
            if setup_script:
                parts.append(f"- Setup script: `{setup_script}` (run before tests)")
        else:
            parts.append("- No tests configured. Generate code and verify with compile_check.")

        parts.append("")

        if rib_provided:
            parts.append("## Architecture (Pre-provided)\n")
            parts.append("A ground-truth RIB has been pre-loaded at `architecture/rib.json`.")
            parts.append("Do NOT call generate_architecture or judge_architecture.")
            parts.append("Read this file directly to get the file list.\n")
        else:
            parts.append("## Tool Parameters\n")
            parts.append("When calling generate_architecture, use these paths:")
            parts.append(f"- prd_path: `{prd_path}`")
            parts.append(f"- uml_class_path: `{uml_class_path}`")
            parts.append(f"- uml_sequence_path: `{uml_sequence_path}`")
            parts.append(f"- arch_design_path: `{arch_design_path}`")
            parts.append(f"- output_path: `architecture/rib.json`")
            parts.append("")

        parts.append(f"Workspace directory (for tools that need it): `.`")
        parts.append("")

        # Workflow guidance
        parts.append("## Required Workflow\n")
        step = 1
        parts.append(f"{step}. **Read** the requirement documents to understand the project")
        step += 1
        if not rib_provided:
            parts.append(f"{step}. **generate_architecture** → **judge_architecture**")
            parts.append("   - If score < 8: edit RIB JSON with file_editor, then **MUST call judge_architecture again**")
            parts.append("   - Repeat edit → re-judge until score ≥ 8 or 3 judge attempts used")
            step += 1
        parts.append(f"{step}. Read RIB to get the file list, then for each file:")
        parts.append("   **generate_rib** → **compile_check**")
        step += 1
        parts.append(f"{step}. **judge_rib** on all skeletons")
        parts.append("   - If score < 8: fix with file_editor, then **MUST call judge_rib again**")
        parts.append("   - Repeat fix → re-judge until score ≥ 8 or 3 judge attempts used")
        step += 1
        parts.append(f"{step}. For each file: **generate_code** → **compile_check**")
        step += 1
        test_step = step
        parts.append(f"{step}. **run_tests** → if failures, read the test output, analyze errors,")
        parts.append("   fix code with apply_patch, then re-run tests")
        step += 1
        parts.append(f"{step}. Repeat step {test_step} until all tests pass or {self.cfg.iteration.testfix_iters} test rounds used")
        step += 1
        parts.append(f"{step}. **Once ALL tests pass, IMMEDIATELY finish.** Do NOT perform any additional")
        parts.append("   cleanup, restructuring, or polishing after tests pass. Call finish right away.")
        parts.append("")

        parts.append("## Goal\n")
        parts.append("Generate a complete, working Python project that passes all tests.")
        parts.append("Start by reading the requirement documents, then follow the required workflow.")
        parts.append("Stop immediately once all tests pass.")

        return "\n".join(parts)

    def _build_summary(self, runtime: CodebaseRuntime, elapsed: float, tracker=None) -> None:
        """Write run summary as JSON with metrics and optional per-stage timing."""
        import json as _json
        metrics = runtime.get_llm_metrics()

        summary: dict = {
            "run_id": self.cfg.run_id,
            "dataset": self.cfg.dataset,
            "repo": self.cfg.repo,
            "model": self.cfg.runtime.model_name,
            "duration_sec": round(elapsed, 1),
            "metrics": {
                "total_cost": round(metrics["total_cost"], 4),
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
            },
        }

        if tracker:
            summary_data = tracker.get_summary()
            cache_read = summary_data.get("total_cache_read_tokens", 0)
            cache_create = summary_data.get("total_cache_creation_tokens", 0)
            prompt_total = summary_data.get("total_prompt_tokens", 0) or metrics["prompt_tokens"]
            cache_rate = (cache_read / prompt_total * 100) if prompt_total else 0

            summary["metrics"]["cache_read_tokens"] = cache_read
            summary["metrics"]["cache_creation_tokens"] = cache_create
            summary["metrics"]["cache_hit_rate"] = round(cache_rate, 1)
            summary["main_iterations"] = summary_data.get("main_iterations", 0)
            summary["phase_summary"] = summary_data.get("phase_summary", {})
            summary["tool_counts"] = summary_data.get("tool_counts", {})
            summary["tool_error_counts"] = summary_data.get("tool_error_counts", {})
            summary["test_results"] = summary_data.get("test_results")

        self.artifact_logger.write_text(
            "summary.json",
            _json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
