"""Codebase-specific config — re-exports common config + autonomous settings."""
import os

from common.config import (  # noqa: F401
    PROJECT_ROOT,
    DATASET_ROOT,
    WORKSPACE_ROOT,
    WORKSPACE_INTERNAL_DIR,
    WORKSPACE_ARCH_SUBDIR,
    IterationConfig,
    RuntimeConfig,
    RunConfig,
    is_bedrock_model,
    get_model_name,
    get_api_key,
    get_base_url,
    get_aws_region,
)

# Sub-agent tool configs for generative tool executors
SUB_AGENT_TOOLS: dict[str, list[str]] = {
    "architecture": ["terminal", "file_editor", "task_tracker", "glob", "grep"],
    "rib":     ["terminal", "file_editor", "apply_patch", "glob", "grep"],
    "code":         ["terminal", "file_editor", "apply_patch", "glob", "grep"],
}

# Maximum iterations for the main agent's conversation
MAX_ITERATION_PER_RUN = 1000

# Progress tracking: enable for debugging, disable for experiments.
# Set ENABLE_PROGRESS_TRACKING=1 in env to activate.
ENABLE_PROGRESS_TRACKING = os.environ.get("ENABLE_PROGRESS_TRACKING", "0") == "1"
