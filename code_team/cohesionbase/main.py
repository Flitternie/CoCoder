"""CLI entry point for cohesionbase pipeline."""
from __future__ import annotations

import argparse
import os
import sys


def main():
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from lmnr import Laminar
        Laminar.initialize(project_api_key=os.environ.get("LMNR_PROJECT_API_KEY", ""))
    except Exception:
        pass

    from common.sdk_patches import apply_all
    apply_all()

    parser = argparse.ArgumentParser(description="cohesionbase: cohesion-based parallel code generation")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the cohesionbase pipeline")
    run_parser.add_argument("--dataset", required=True, help="Dataset name (e.g. CodeProjectEval)")
    run_parser.add_argument("--repo", required=True, help="Repository name within the dataset")
    run_parser.add_argument("--model", default=None, help="Model name override")
    run_parser.add_argument("--rib-file", nargs="?", const=True, default=False,
                           help="Use ground-truth RIB file (skip architecture generation). "
                                "No value = default dataset path; or specify a custom file path.")
    run_parser.add_argument("--rib-dep-tool", action="store_true", default=False,
                           help="Architecture generation includes LLM-inferred dependencies")
    run_parser.add_argument("--no-visualizer", action="store_true", default=False,
                           help="Disable conversation visualizer (for parallel/batch runs)")

    list_parser = subparsers.add_parser("list", help="List available repositories")
    list_parser.add_argument("--dataset", required=True, help="Dataset name")

    args = parser.parse_args()

    if args.command == "run":
        from cohesionbase.pipeline import CohesionPipeline

        run_config = CohesionPipeline.create_run_config(
            dataset=args.dataset,
            repo=args.repo,
            model_name=args.model,
            gt_rib=args.rib_file,
            rib_dep_tool=args.rib_dep_tool,
            no_visualizer=args.no_visualizer,
        )
        pipeline = CohesionPipeline(run_config)
        pipeline.run()

    elif args.command == "list":
        from common.datasets import list_repos
        repos = list_repos(args.dataset)
        for repo in repos:
            print(repo)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
