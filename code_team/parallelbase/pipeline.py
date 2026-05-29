"""ParallelPipeline — load data → build workspace → orchestrate multi-agent run.

Mirrors CodebasePipeline structure but delegates to ParallelOrchestrator
instead of a single-agent runtime.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import time

from common.infra.artifact_logger import ArtifactLogger
from common.datasets import load_repo_bundle, get_numbered_repo_name
from common.models import RepoBundle
from common.workspace_builder import build_workspace, copy_tests_for_final_eval
from parallelbase.prompts import build_leader_user_prompt

from common.config import (
    IterationConfig,
    RunConfig,
    RuntimeConfig,
    get_model_name,
    get_api_key,
    get_base_url,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path("workspaces")


class ParallelPipeline:
    """Multi-agent parallel pipeline: Leader + Architecture + Code×N."""

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
        no_visualizer: bool = False,
    ) -> RunConfig:
        import re

        model = get_model_name(model_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_slug = model.rsplit("/", 1)[-1]
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_slug).strip("._-") or "model"
        dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._-") or "dataset"
        repo_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo).strip("._-") or "repo"

        if gt_rib:
            rib_tag = "ribfile"
        else:
            rib_tag = "ribgensim"
        run_id = f"{timestamp}_{model_slug}_parallelbase_{rib_tag}_{dataset_slug}_{repo_slug}"
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
            iteration=IterationConfig(),
            gt_rib=gt_rib,
            no_visualizer=no_visualizer,
        )

    def run(self) -> None:
        """Execute the parallel pipeline."""
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

        # Save run metadata
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
            "pipeline": "parallelbase",
            "model": self.cfg.runtime.model_name,
        }
        self.artifact_logger.write_json("meta.json", meta)

        # Optional progress tracker
        tracker = None
        enable_tracking = os.environ.get("ENABLE_PROGRESS_TRACKING", "0") == "1"
        if enable_tracking:
            from dashboard.parallelbase.parallel_progress import ParallelProgressTracker
            tracker = ParallelProgressTracker(self.run_dir, self.cfg.run_id)
            tracker.start()

        # Create orchestrator
        from parallelbase.orchestrator import ParallelOrchestrator

        orchestrator = ParallelOrchestrator(
            workspace_dir=workspace_dir,
            run_dir=self.run_dir,
            runtime_cfg=self.cfg.runtime,
            tracker=tracker,
            testfix_iters=self.cfg.iteration.testfix_iters,
            no_visualizer=self.cfg.no_visualizer,
        )

        tracker_stopped = False
        try:
            # Run setup script if configured
            setup_script = bundle.config.get("setup_shell_script", "")
            if setup_script:
                import subprocess
                script_path = workspace_dir / setup_script
                if script_path.exists():
                    print(f"[parallelbase] Running setup script: {setup_script}")
                    subprocess.run(
                        ["bash", str(script_path)],
                        cwd=str(workspace_dir),
                        timeout=300,
                        capture_output=True,
                    )

            # If ground-truth RIB requested, copy into workspace
            from common.config import copy_gt_rib_to_workspace, resolve_gt_rib
            rib_provided = copy_gt_rib_to_workspace(self.cfg, workspace_dir, default_filename="rib_dep_ground_true.json")
            if rib_provided:
                print(f"[parallelbase] Using ground-truth RIB: {resolve_gt_rib(self.cfg, 'rib_dep_ground_true.json')}")

            # Build the initial prompt for Leader
            prompt = self._build_initial_prompt(bundle, workspace_dir, rib_provided=rib_provided)
            self.artifact_logger.write_text("artifacts/initial_prompt.txt", prompt)

            # Start the orchestrator (blocks until done)
            orchestrator.start(prompt)

            if tracker:
                try:
                    metrics = orchestrator.get_llm_metrics()
                    tracker.update_llm_metrics(
                        total_cost=metrics["total_cost"],
                        prompt_tokens=metrics["prompt_tokens"],
                        completion_tokens=metrics["completion_tokens"],
                    )
                except Exception:
                    pass  # Fall back to file-scanned metrics
                tracker.stop()
                tracker_stopped = True

            # Save summary
            elapsed = time.time() - start
            self._build_summary(orchestrator, elapsed, tracker)

        finally:
            # Inject authoritative LLM metrics before final flush
            if tracker and not tracker_stopped:
                try:
                    metrics = orchestrator.get_llm_metrics()
                    tracker.update_llm_metrics(
                        total_cost=metrics["total_cost"],
                        prompt_tokens=metrics["prompt_tokens"],
                        completion_tokens=metrics["completion_tokens"],
                    )
                except Exception:
                    pass  # Fall back to file-scanned metrics
                tracker.stop()
            orchestrator.close()

    def _build_initial_prompt(self, bundle: RepoBundle, workspace_dir: Path, *, rib_provided: bool = False) -> str:
        """Build the initial prompt for the Main Agent."""
        return build_leader_user_prompt(bundle, testfix_iters=self.cfg.iteration.testfix_iters, rib_provided=rib_provided)

    def _build_summary(self, orchestrator, elapsed: float, tracker=None) -> None:
        """Write run summary as JSON with metrics."""
        import json as _json
        metrics = orchestrator.get_llm_metrics()

        summary: dict = {
            "run_id": self.cfg.run_id,
            "dataset": self.cfg.dataset,
            "repo": self.cfg.repo,
            "model": self.cfg.runtime.model_name,
            "pipeline": "parallelbase",
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
            summary["agents"] = summary_data.get("agents", {})

        self.artifact_logger.write_text(
            "summary.json",
            _json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
