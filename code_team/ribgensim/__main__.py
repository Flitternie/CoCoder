#!/usr/bin/env python3
"""CLI entry point for RIB generation (simple, generate-only).

Usage:
    # Default (no deps)
    python -m ribgensim --dataset DevEval --repo readtime

    # With dependency inference
    python -m ribgensim --dataset DevEval --repo readtime --rib-dep-tool

    # Override model
    python -m ribgensim --dataset DevEval --repo readtime --model bedrock/moonshotai.kimi-k2.5

    # Multiple repos
    python -m ribgensim --dataset DevEval --repo readtime chakin geotext

    # All repos in dataset
    python -m ribgensim --dataset DevEval
"""
from __future__ import annotations

import argparse

from ribgensim import run


def main():
    parser = argparse.ArgumentParser(
        prog="ribgensim",
        description="Generate RIB architecture (simple, no judge loop)",
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Dataset name, e.g. DevEval or CodeProjectEval",
    )
    parser.add_argument(
        "--repo", "-r",
        nargs="*",
        default=None,
        help="Repo name(s). If omitted, runs all repos in the dataset",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name override (default: RIB_DEP_MODEL if --rib-dep-tool, else LLM_MODEL)",
    )
    parser.add_argument(
        "--rib-dep-tool",
        action="store_true",
        default=False,
        help="Use architecture prompt with dependency inference",
    )
    args = parser.parse_args()

    run.run(
        dataset=args.dataset,
        repos=args.repo,
        model_override=args.model,
        use_dep=args.rib_dep_tool,
    )


if __name__ == "__main__":
    main()
