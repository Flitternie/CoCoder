"""RIB generation pipeline — generate-only (no judge/fix loop).

For each project, creates an AgentRuntime, runs the architecture generation
sub-agent once (with or without dependencies), and saves the resulting rib.json.

Supports:
  - --rib-dep-tool: use GenerateArchitectureDepTool prompt (with dependencies)
  - --model: override the generation model (or reads LLM_MODEL / RIB_DEP_MODEL)
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_TEAM_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_TEAM_DIR.parent

sys.path.insert(0, str(CODE_TEAM_DIR))

from common.sdk_patches import apply_all as _apply_sdk_patches
_apply_sdk_patches()

from common.config import (
    RuntimeConfig,
    get_model_name,
    get_api_key,
    get_base_url,
    get_rib_dep_runtime_cfg,
    is_bedrock_model,
)
from common.datasets import load_repo_bundle, list_repos, get_numbered_repo_name
from common.prompt_constants import load_implementation_integrity_rules
from common.prompts import architecture_prompt, architecture_prompt_dep
from common.workspace_builder import build_workspace, copy_tests_for_final_eval
from common.agent_runtime import AgentRuntime
from codebase.config import SUB_AGENT_TOOLS


# ---------------------------------------------------------------------------
# Prompt builders (reuse the same prompts as the actual tools)
# ---------------------------------------------------------------------------

def _build_prompt(bundle, workspace_dir: Path, *, use_dep: bool) -> str:
    """Build the architecture generation prompt from requirement docs."""
    cfg = bundle.config
    prd_path = cfg.get("PRD", "")
    uml_class_path = cfg.get("UML_class", "")
    if not uml_class_path:
        uml_list = cfg.get("UML", [])
        if isinstance(uml_list, list) and uml_list:
            for item in uml_list:
                if "pyreverse" in item:
                    uml_class_path = item
                    break
            if not uml_class_path:
                uml_class_path = uml_list[0]
    uml_seq_path = cfg.get("UML_sequence", "")
    arch_design_path = cfg.get("architecture_design", "")

    def _read(p: str) -> str:
        return (workspace_dir / p).read_text(encoding="utf-8") if p else ""

    prd = _read(prd_path)
    uml_class = _read(uml_class_path)
    uml_seq = _read(uml_seq_path)
    arch_design = _read(arch_design_path)

    output_path = str(workspace_dir / "architecture" / "rib.json")

    pre_existing = sorted(
        str(p.relative_to(workspace_dir))
        for p in workspace_dir.rglob("*")
        if p.is_file()
    )

    integrity_rules = load_implementation_integrity_rules(workspace_dir)

    if use_dep:
        return architecture_prompt_dep(
            requirement=prd,
            uml_class=uml_class,
            uml_sequence=uml_seq,
            arch_design=arch_design,
            output_json_path=output_path,
            pre_existing_files=pre_existing,
            integrity_rules=integrity_rules,
        )
    else:
        return architecture_prompt(
            requirement=prd,
            uml_class=uml_class,
            uml_sequence=uml_seq,
            arch_design=arch_design,
            output_json_path=output_path,
            pre_existing_files=pre_existing,
            integrity_rules=integrity_rules,
        )


# ---------------------------------------------------------------------------
# Per-project
# ---------------------------------------------------------------------------

def process_project(
    dataset: str,
    repo: str,
    model_name: str,
    run_id: str,
    *,
    use_dep: bool = False,
    runtime_cfg_override: RuntimeConfig | None = None,
) -> dict:
    project_name = f"{dataset}/{repo}"

    bundle = load_repo_bundle(dataset, repo)
    cfg = bundle.config

    numbered_repo = get_numbered_repo_name(dataset, repo)
    result_base = CODE_TEAM_DIR / "results" / "ribgensim" / dataset / numbered_repo / run_id
    run_dir = result_base / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = result_base / "workspaces"

    manifest = build_workspace(bundle, workspace_dir)
    print(f"  Workspace built: {workspace_dir} ({len(manifest.copied)} files)")

    if cfg.get("check_tests"):
        copy_tests_for_final_eval(bundle, workspace_dir, "check_tests")

    prompt = _build_prompt(bundle, workspace_dir, use_dep=use_dep)

    if runtime_cfg_override is not None:
        runtime_cfg = runtime_cfg_override
    else:
        base_url = None if is_bedrock_model(model_name) else get_base_url()
        runtime_cfg = RuntimeConfig(
            model_name=model_name,
            api_key=get_api_key(model_name),
            base_url=base_url,
        )

    runtime = AgentRuntime(
        workspace_dir,
        run_dir,
        runtime_cfg,
        agent_tool_configs=SUB_AGENT_TOOLS,
    )

    t0 = time.time()
    try:
        runtime.run_agent("architecture", prompt, timeout_sec=600)
        elapsed = time.time() - t0
        print(f"  Sub-agent finished in {elapsed:.1f}s")

        rib_path = workspace_dir / "architecture" / "rib.json"
        rib_exists = rib_path.exists()
        module_count = 0
        file_count = 0
        if rib_exists:
            data = json.loads(rib_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = [data]
            module_count = len(data)
            file_count = sum(len(m.get("files", [])) for m in data)
            print(f"  RIB: {module_count} modules, {file_count} files")
        else:
            print(f"  WARNING: rib.json not found")

        metrics = runtime.get_llm_metrics()
        print(f"  Cost: ${metrics['total_cost']:.4f}")

        result = {
            "project": project_name,
            "elapsed": round(elapsed, 1),
            "cost": round(metrics["total_cost"], 4),
            "rib_exists": rib_exists,
            "modules": module_count,
            "files": file_count,
            "use_dep": use_dep,
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {e}")
        result = {"project": project_name, "error": str(e), "elapsed": round(elapsed, 1)}
    finally:
        runtime.close()

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Summary: {summary_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    dataset: str,
    repos: list[str] | None = None,
    model_override: str | None = None,
    *,
    use_dep: bool = False,
):
    load_dotenv(PROJECT_ROOT / ".env")

    runtime_cfg_override: RuntimeConfig | None = None
    if use_dep and model_override is None:
        runtime_cfg_override = get_rib_dep_runtime_cfg()
        if runtime_cfg_override is not None:
            model_name = runtime_cfg_override.model_name
        else:
            model_name = get_model_name()
    else:
        model_name = get_model_name(override=model_override)

    print(f"Model: {model_name}")
    print(f"Dep mode: {'ON' if use_dep else 'OFF'}")

    if repos:
        repo_list = repos
    else:
        repo_list = list_repos(dataset)
    print(f"Dataset: {dataset}, Repos: {len(repo_list)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.rsplit("/", 1)[-1]
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_slug).strip("._-") or "model"
    dep_tag = "_dep" if use_dep else ""
    run_id = f"{timestamp}_{model_slug}{dep_tag}_ribgensim"

    results = []
    for repo in repo_list:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}/{repo}")
        print(f"{'='*60}")

        result = process_project(dataset, repo, model_name, run_id, use_dep=use_dep, runtime_cfg_override=runtime_cfg_override)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r.get("rib_exists") else "FAIL"
        print(f"  [{status}] {r['project']}: {r.get('elapsed', '?')}s, ${r.get('cost', '?')}")

    total_cost = sum(r.get("cost", 0) for r in results)
    print(f"\nTotal cost: ${total_cost:.4f}")

    return results
