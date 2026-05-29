# Claude Code Agent Team

This pipeline runs `claude` CLI as a subprocess and routes requests through a local [LiteLLM](https://docs.litellm.ai/) proxy, which supports 100+ providers (OpenAI, Anthropic, Bedrock, Azure, Portkey, etc.).

All LLM settings are read from the project root `.env` file, same as the `codebase` pipeline.

## Setup

### 1. Configure `.env`

Edit the project root `.env` to set your model and provider:

```bash
# Portkey + OpenAI
LLM_MODEL=openai/gpt-5-mini
LLM_API_KEY=your-portkey-key
LLM_BASE_URL=https://api.portkey.ai/v1
PORTKEY_PROVIDER=@your-provider-id

# Portkey + Anthropic (anthropic/ prefix uses base URL without /v1)
LLM_MODEL=anthropic/claude-sonnet-4-5-20250929
LLM_API_KEY=your-portkey-key
LLM_BASE_URL=https://api.portkey.ai
PORTKEY_PROVIDER=@your-provider-id

# Direct OpenAI (no proxy gateway)
LLM_MODEL=openai/gpt-5-mini
LLM_API_KEY=sk-...
```

### 2. Configure `.claude/settings.json`

The project-level `.claude/settings.json` (already included in this directory) points Claude Code at the local proxy. If missing, create it:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://0.0.0.0:4000",
    "ANTHROPIC_AUTH_TOKEN": "sk-litellm-benchmarking"
  }
}
```

### 3. Start the proxy

```bash
cd code_team/claude_code_agent_team
source start_proxy.sh
```

This loads `.env`, starts a LiteLLM proxy on `http://0.0.0.0:4000`, and sets the Claude model aliases to route through the proxy.

### 4. Run the pipeline

```bash
# Uses default RIB from datasets/depanalysis/{dataset}/{repo}/
python -m claude_code_agent_team run --dataset CodeProjectEval --repo pyjwt

# Custom RIB file
python -m claude_code_agent_team run --dataset CodeProjectEval --repo pyjwt --rib-file path/to/rib.json

# List available repos
python -m claude_code_agent_team list --dataset CodeProjectEval
```

### 5. Stop the proxy

```bash
stop_proxy
# or manually:
lsof -ti:4000 | xargs kill -9
```
