#!/usr/bin/env python3
"""Claude Code agent team pipeline CLI.

Usage:
    python -m claude_code_agent_team run --dataset CodeProjectEval --repo pyjwt
    python -m claude_code_agent_team run --dataset CodeProjectEval --repo pyjwt --rib-file path/to/rib.json
    python -m claude_code_agent_team list --dataset CodeProjectEval
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude_code_agent_team",
        description="Code generation pipeline using Claude Code agent teams",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run claude_code_agent_team pipeline")
    run_p.add_argument("--dataset", required=True)
    run_p.add_argument("--repo", required=True)
    run_p.add_argument(
        "--model",
        default=None,
        help="Claude model alias or ID (env: CLAUDE_CODE_MODEL, default: sonnet)",
    )
    run_p.add_argument(
        "--rib-file",
        default=True,
        nargs="?",
        const=True,
        help="RIB file path (default: auto-detect from dataset). Required.",
    )

    list_p = sub.add_parser("list", help="List repos in a dataset")
    list_p.add_argument("--dataset", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        from common.datasets import list_repos_numbered
        for repo in list_repos_numbered(args.dataset):
            print(repo)
        return

    if args.command == "run":
        from claude_code_agent_team.pipeline import ClaudeCodePipeline

        run_config = ClaudeCodePipeline.create_run_config(
            dataset=args.dataset,
            repo=args.repo,
            model=args.model,
            gt_rib=args.rib_file,
        )
        pipeline = ClaudeCodePipeline(run_config)
        pipeline.run()
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
