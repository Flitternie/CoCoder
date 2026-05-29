#!/usr/bin/env python3
"""Codebase agent CLI entry point.

Usage:
    python -m codebase run --dataset DevEval --repo readtime [OPTIONS]
    python -m codebase list --dataset DevEval
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

# Laminar integration (optional)
LMNR_KEY = os.environ.get("LMNR_PROJECT_API_KEY", "").strip()
if LMNR_KEY:
    try:
        from lmnr import Laminar
        Laminar.initialize(project_api_key=LMNR_KEY)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebase",
        description="Autonomous agent: 1 agent + 7 custom tools, self-directed workflow",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Run codebase agent from scratch")
    run_p.add_argument("--dataset", required=True)
    run_p.add_argument("--repo", required=True)
    run_p.add_argument("--model", default=None, help="LLM model name (env: LLM_MODEL)")
    run_p.add_argument("--rib-file", nargs="?", const=True, default=False,
                       help="Use ground-truth RIB file (skip architecture generation). "
                            "No value = default dataset path; or specify a custom file path.")
    run_p.add_argument("--rib-dep-tool", action="store_true", default=False,
                       help="Architecture generation includes LLM-inferred dependencies")
    run_p.add_argument("--no-visualizer", action="store_true", default=False,
                       help="Disable conversation visualizer (for parallel/batch runs)")

    # --- list ---
    list_p = sub.add_parser("list", help="List repos in the dataset")
    list_p.add_argument("--dataset", required=True)

    return parser


def main() -> None:
    # Apply SDK monkey-patches before anything else
    from common.sdk_patches import apply_all as _apply_sdk_patches
    _apply_sdk_patches()

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        from common.datasets import list_repos_numbered
        repos = list_repos_numbered(args.dataset)
        for repo_name in repos:
            print(repo_name)
        return

    if args.command == "run":
        from codebase.pipeline import CodebasePipeline

        run_config = CodebasePipeline.create_run_config(
            dataset=args.dataset,
            repo=args.repo,
            model_name=args.model,
            gt_rib=args.rib_file,
            rib_dep_tool=args.rib_dep_tool,
            no_visualizer=args.no_visualizer,
        )
        pipeline = CodebasePipeline(run_config)
        pipeline.run()
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
