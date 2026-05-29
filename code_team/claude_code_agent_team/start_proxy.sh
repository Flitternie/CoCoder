#!/bin/bash
# start_proxy.sh
# Starts the LiteLLM proxy for routing Claude CLI requests through litellm.
# Usage: source start_proxy.sh
#
# All LLM settings are read from the project root .env file.
# litellm_config.yaml references them via os.environ/ syntax.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Load .env ─────────────────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo "Loaded $PROJECT_ROOT/.env"
else
    echo "ERROR: $PROJECT_ROOT/.env not found"
    return 1 2>/dev/null || exit 1
fi

: "${LLM_MODEL:?LLM_MODEL not set in .env}"

# ── Defaults for optional vars (avoid os.environ/ lookup failure) ─────────────
export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_BASE_URL="${LLM_BASE_URL:-}"
export PORTKEY_PROVIDER="${PORTKEY_PROVIDER:-}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-litellm-benchmarking}"

# ── Proxy connection (ensure Claude CLI routes through local proxy) ────────────
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://0.0.0.0:4000}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$LITELLM_MASTER_KEY}"

# ── Model routing (override claude model names to route through proxy) ────────
export CLAUDE_CODE_MODEL="$LLM_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$LLM_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$LLM_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$LLM_MODEL"

# ── Enable Claude Code agent teams (experimental) ────────────────────────────
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# ── Start proxy ──────────────────────────────────────────────────────────────
echo "Starting LiteLLM proxy: model=$LLM_MODEL port=4000"
litellm --config "$SCRIPT_DIR/litellm_config.yaml" --port 4000 &
LITELLM_PID=$!
echo "LiteLLM proxy started (PID $LITELLM_PID) on http://0.0.0.0:4000"

sleep 2
curl -s http://0.0.0.0:4000/health > /dev/null && echo "Proxy is healthy" || echo "WARNING: proxy not responding"

# ── Convenience ───────────────────────────────────────────────────────────────
stop_proxy() {
    lsof -ti:4000 | xargs kill -9 && echo "Proxy stopped"
}

echo ""
echo "Model: $LLM_MODEL"
echo "Commands: stop_proxy"
