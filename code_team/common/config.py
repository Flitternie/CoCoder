from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import litellm
from common.bedrock_mantle import MantleProxy, ensure_mantle_proxy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "datasets"
WORKSPACE_ROOT = Path("workspaces")
WORKSPACE_INTERNAL_DIR = "tmp_files"
WORKSPACE_ARCH_SUBDIR = "architecture"
TERMINAL_NO_CHANGE_TIMEOUT_SECONDS = int(os.environ.get("TERMINAL_NO_CHANGE_TIMEOUT_SECONDS", "15"))


@dataclass(slots=True)
class IterationConfig:
    arch_iters: int = 3
    skeleton_iters: int = 3
    testfix_iters: int = 10


@dataclass(slots=True)
class RuntimeConfig:
    model_name: str = ""
    api_key: str = ""
    base_url: str | None = None
    num_retries: int = 3
    max_workers: int = 35
    max_output_tokens: int = 64000
    llm_call_timeout_sec: int = 300
    compile_timeout_sec: int = 600


@dataclass(slots=True)
class RunConfig:
    dataset: str
    repo: str
    run_id: str
    run_dir: Path
    workspace_dir: Path
    iteration: IterationConfig
    runtime: RuntimeConfig
    gt_rib: str | bool = False  # False=off, True=default dataset path, str=custom file path
    rib_dep_tool: bool = False  # True=use architecture with dependencies, False=codebase version without deps
    no_visualizer: bool = False


def resolve_gt_rib(cfg: RunConfig, default_filename: str = "rib_ground_true.json") -> Path | None:
    """Resolve the ground-truth RIB source path from RunConfig.gt_rib.

    Returns None if gt_rib is False (disabled).
    Returns the resolved Path if gt_rib is True (default dataset path) or a string (custom path).
    Raises FileNotFoundError if the resolved path does not exist.
    """
    if not cfg.gt_rib:
        return None

    if isinstance(cfg.gt_rib, str):
        src = Path(cfg.gt_rib).resolve()
    else:
        # Default: datasets/depanalysis/{dataset}/{repo}/{default_filename}
        src = DATASET_ROOT / "depanalysis" / cfg.dataset / cfg.repo / default_filename

    if not src.exists():
        raise FileNotFoundError(
            f"Ground-truth RIB not found: {src}\n"
            f"Provide a custom path via --rib-file <path>, or disable RIB if the pipeline enables it by default."
        )
    return src


def copy_gt_rib_to_workspace(cfg: RunConfig, workspace_dir: Path, default_filename: str = "rib_ground_true.json") -> bool:
    """Copy ground-truth RIB into workspace/architecture/rib.json if configured.

    Returns True if RIB was copied, False if gt_rib is disabled.
    """
    src = resolve_gt_rib(cfg, default_filename)
    if src is None:
        return False

    import shutil
    dst = workspace_dir / "architecture" / "rib.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def is_bedrock_model(model_name: str) -> bool:
    """Check whether the model is an AWS Bedrock model (prefixed with 'bedrock/')."""
    return isinstance(model_name, str) and model_name.lower().startswith("bedrock/")


def get_sampling_params(model_name: str) -> dict[str, float]:
    """Return sampling parameters (temperature, top_p) appropriate for the model.

    Kimi K2.5 only accepts temperature=1 and top_p=0.95.
    All other models default to temperature=0, top_p=1 for deterministic output.
    """
    name = model_name.lower() if isinstance(model_name, str) else ""
    if "kimi-k2.5" in name:
        return {"temperature": 1, "top_p": 0.95}
    return {"temperature": 0, "top_p": 1}


# ---------------------------------------------------------------------------
# Kimi K2.5 + Bedrock Mantle: module-level proxy state
# ---------------------------------------------------------------------------
_mantle_proxy: MantleProxy | None = None


def _is_bedrock_kimi(model: str) -> bool:
    low = model.lower() if isinstance(model, str) else ""
    return low.startswith("bedrock/") and "kimi-k2.5" in low


def _ensure_mantle_proxy() -> None:
    """Start the local SigV4 proxy for Bedrock Mantle (once).

    Kimi K2.5 has tool-call parsing bugs via Bedrock Converse API
    (premature end_turn, leaked tokens, empty responses).  Routing
    through the Mantle Chat Completions endpoint avoids Converse.
    """
    global _mantle_proxy
    if _mantle_proxy is not None:
        return
    _mantle_proxy = ensure_mantle_proxy(get_aws_region())

    # Register pricing so LiteLLM can calculate costs for the remapped model.
    # After proxy mapping the model appears as "openai/moonshotai.kimi-k2.5",
    # which isn't in LiteLLM's built-in pricing database.
    litellm.register_model({
        "openai/moonshotai.kimi-k2.5": {
            "max_tokens": 262144,
            "max_input_tokens": 262144,
            "max_output_tokens": 262144,
            "input_cost_per_token": 7.3e-07,
            "output_cost_per_token": 3.03e-06,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    })


# Override Bedrock pricing to match Anthropic API pricing,
# so cost comparisons across providers are apples-to-apples.
def _align_bedrock_pricing_to_anthropic() -> None:
    _PRICING_KEYS = ("input_cost_per_token", "output_cost_per_token",
                     "cache_read_input_token_cost", "cache_creation_input_token_cost")
    for model_id, info in list(litellm.model_cost.items()):
        if not any(model_id.startswith(p) for p in ("anthropic.", "us.anthropic.", "eu.anthropic.",
                                                     "ap.anthropic.", "jp.anthropic.", "us-gov.anthropic.",
                                                     "au.anthropic.", "global.anthropic.")):
            continue
        # e.g. "us.anthropic.claude-sonnet-4-5-20250929-v1:0" -> "claude-sonnet-4-5-20250929"
        base = model_id.split(".", 2)[-1]                       # "claude-sonnet-4-5-20250929-v1:0"
        base = base.rsplit("-v", 1)[0] if "-v1:" in base else base  # "claude-sonnet-4-5-20250929"
        anthropic_info = litellm.model_cost.get(base, {})
        if not anthropic_info:
            continue
        for k in _PRICING_KEYS:
            if k in anthropic_info:
                info[k] = anthropic_info[k]

_align_bedrock_pricing_to_anthropic()


def get_portkey_kwargs(model: str) -> dict[str, object]:
    """Build Portkey-related LLM kwargs (extra_headers).

    Returns a dict that can be **unpacked into LLM(...) kwargs.
    Returns empty dict when Portkey is not configured.
    """
    provider = os.environ.get("PORTKEY_PROVIDER", "").strip()
    config = os.environ.get("PORTKEY_CONFIG", "").strip()
    if not provider and not config:
        return {}
    portkey_headers: dict[str, str] = {}
    if provider:
        portkey_headers["x-portkey-provider"] = provider
    if config:
        portkey_headers["x-portkey-config"] = config
    if model.startswith("anthropic/"):
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        if api_key:
            portkey_headers["x-portkey-api-key"] = api_key
    return {"extra_headers": portkey_headers}


def get_model_name(override: str | None = None) -> str:
    """Get the model name: CLI override > environment variable LLM_MODEL."""
    if override:
        model = override
    else:
        model = os.environ.get("LLM_MODEL", "").strip()
        if not model:
            raise ValueError("Model not specified: please set via --model argument or LLM_MODEL environment variable.")
            # No model specified: set via --model flag or LLM_MODEL env var.
    if _is_bedrock_kimi(model):
        _ensure_mantle_proxy()
        # Tell LiteLLM it's a plain OpenAI endpoint; the proxy handles SigV4.
        return "openai/" + model[len("bedrock/"):]
    return model


def get_api_key(model_name: str) -> str:
    """Get the API key: Bedrock requires no key; others read LLM_API_KEY."""
    if _mantle_proxy is not None:
        return "bedrock-sigv4"  # dummy; proxy does real auth
    if is_bedrock_model(model_name):
        return ""
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Environment variable LLM_API_KEY is not set.")
        # LLM_API_KEY env var not set.
    return api_key


def get_base_url() -> str | None:
    """Get the LLM base URL (optional)."""
    if _mantle_proxy is not None:
        return _mantle_proxy.base_url
    url = os.environ.get("LLM_BASE_URL", "").strip()
    return url or None


def get_aws_region(default: str = "us-east-1") -> str:
    """Get the AWS region, preferring AWS_REGION / AWS_REGION_NAME env vars."""
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_REGION_NAME") or default


def get_rib_dep_model_slug() -> str:
    """Short name of RIB_DEP_MODEL for run-id tagging. Empty if unset."""
    import re
    raw = os.environ.get("RIB_DEP_MODEL", "").strip()
    if not raw:
        return ""
    slug = raw.rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug).strip("._-")
    return slug or "model"


def get_rib_dep_runtime_cfg() -> RuntimeConfig | None:
    """Build a RuntimeConfig for the RIB dep tool if RIB_DEP_MODEL is set.

    Returns None when the env var is empty/unset (caller should use default runtime).
    Reads RIB_DEP_API_KEY for the API key (bedrock models need no key).
    """
    raw = os.environ.get("RIB_DEP_MODEL", "").strip()
    if not raw:
        return None
    uses_kimi_proxy = _is_bedrock_kimi(raw)
    model = get_model_name(override=raw)
    if uses_kimi_proxy:
        api_key = "bedrock-sigv4"
        base_url = get_base_url()
    elif is_bedrock_model(model):
        api_key = ""
        url = os.environ.get("LLM_BASE_URL", "").strip()
        base_url = url or None
    else:
        api_key = (
            os.environ.get("RIB_DEP_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError("RIB_DEP_API_KEY (or LLM_API_KEY) is required when RIB_DEP_MODEL is a non-Bedrock model.")
        url = (
            os.environ.get("RIB_DEP_BASE_URL", "").strip()
            or os.environ.get("LLM_BASE_URL", "").strip()
        )
        base_url = url or None
    return RuntimeConfig(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
    )
