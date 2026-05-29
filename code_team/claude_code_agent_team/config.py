"""claude_code_agent_team config — thin wrapper around common config."""
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
)

# Default model alias for the claude CLI (sonnet, opus, haiku, or a full model ID)
DEFAULT_CLAUDE_MODEL = "gpt-5-mini"

# Timeout for the full claude subprocess in seconds (2 hours)
CLAUDE_SUBPROCESS_TIMEOUT = 7200

# Progress tracking flag (same env var as codebase pipeline)
ENABLE_PROGRESS_TRACKING = os.environ.get("ENABLE_PROGRESS_TRACKING", "0") == "1"


def get_claude_model(override: str | None = None) -> str:
    """Resolve claude CLI model alias: CLI override > CLAUDE_CODE_MODEL env > default."""
    if override:
        return override
    return os.environ.get("CLAUDE_CODE_MODEL", DEFAULT_CLAUDE_MODEL).strip()
