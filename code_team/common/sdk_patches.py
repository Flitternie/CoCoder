"""One-time monkey-patches for the OpenHands SDK.

Import this module early (e.g. in main) so patches are applied
before any agent or conversation is created.
"""
from threading import Lock


def _patch_action_type_factories() -> None:
    """SDK v1.11.x create_action_type_with_risk / _create_action_type_with_summary
    are not thread-safe — parallel delegate threads can create duplicate
    Pydantic classes.  Wrapping with a single lock makes the
    check-then-create pattern atomic."""
    import openhands.sdk.tool.tool as m
    _lock = Lock()
    _f1, _f2 = m.create_action_type_with_risk, m._create_action_type_with_summary

    def safe_f1(t):
        with _lock:
            return _f1(t)

    def safe_f2(t):
        with _lock:
            return _f2(t)

    m.create_action_type_with_risk = safe_f1
    m._create_action_type_with_summary = safe_f2


def _patch_terminal_timeout_cap(max_seconds: int = 60) -> None:
    """Hard-cap the terminal tool's timeout parameter.

    LLMs sometimes set timeout=120000 (thinking milliseconds) which means
    33 hours.  This clamps any value above *max_seconds* via both __init__
    and model_validate (Pydantic v2 frozen models bypass __init__ when
    constructed through model_validate)."""
    from openhands.tools.terminal.definition import TerminalAction

    _orig_init = TerminalAction.__init__
    _orig_validate = TerminalAction.model_validate.__func__

    def _clamp_timeout(instance):
        if instance.timeout is not None and instance.timeout > max_seconds:
            instance.__dict__["timeout"] = float(max_seconds)

    def _capped_init(self, **kwargs):
        _orig_init(self, **kwargs)
        _clamp_timeout(self)

    @classmethod
    def _capped_validate(cls, *args, **kwargs):
        instance = _orig_validate(cls, *args, **kwargs)
        _clamp_timeout(instance)
        return instance

    TerminalAction.__init__ = _capped_init
    TerminalAction.model_validate = _capped_validate


def _set_tool_text_content_limit(limit: int = 120000) -> None:
    """Raise OpenHands tool text / file_editor response truncation limits.

    The SDK ships with conservative defaults (50K for message-level
    truncation, 16K for file_editor view output).  This patches all
    relevant modules so agents can read large files without surprises.
    """
    if limit <= 0:
        return
    for mod_path, attr in [
        ("openhands.sdk.llm.message", "DEFAULT_TEXT_CONTENT_LIMIT"),
        ("openhands.sdk.utils", "DEFAULT_TEXT_CONTENT_LIMIT"),
        ("openhands.sdk.utils.truncate", "DEFAULT_TEXT_CONTENT_LIMIT"),
        # file_editor has its own response-level truncation (default 16K)
        ("openhands.tools.file_editor.utils.constants", "MAX_RESPONSE_LEN_CHAR"),
        ("openhands.tools.file_editor.editor", "MAX_RESPONSE_LEN_CHAR"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            setattr(mod, attr, limit)
        except Exception:
            pass


def _patch_stream_usage() -> None:
    """Inject ``stream_options={"include_usage": True}`` into all streaming calls.

    Without this, Bedrock ConverseStream omits the final usage metadata chunk,
    so cache_read/cache_creation token counters stay at 0 even though caching
    works on the AWS side.

    Patches the LLM class directly so all instances are covered.
    """
    from openhands.sdk import LLM

    _original = LLM._transport_call

    def _transport_with_stream_usage(self, *, messages, enable_streaming=False, on_token=None, **kwargs):
        if enable_streaming:
            kwargs.setdefault("stream_options", {"include_usage": True})
        return _original(
            self, messages=messages, enable_streaming=enable_streaming,
            on_token=on_token, **kwargs,
        )

    LLM._transport_call = _transport_with_stream_usage


def _is_portkey_openai_claude() -> bool:
    """Check if we're routing a Claude model through Portkey via the openai/ prefix."""
    import os
    model = os.environ.get("LLM_MODEL", "")
    has_portkey = bool(
        os.environ.get("PORTKEY_PROVIDER", "").strip()
        or os.environ.get("PORTKEY_CONFIG", "").strip()
    )
    return has_portkey and model.startswith("openai/") and "claude" in model.lower()


def _patch_preserve_cache_control() -> None:
    """Prevent litellm from stripping cache_control in the OpenAI path.

    Portkey's /v1/chat/completions endpoint supports Anthropic cache_control
    markers, but litellm removes them assuming a standard OpenAI backend.
    """
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

    OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools = (
        lambda self, model, messages, tools=None: (messages, tools)
    )


def _register_portkey_anthropic_pricing() -> None:
    """Register pricing for Anthropic models accessed via openai/ prefix through Portkey.

    Copies the full pricing from litellm's built-in anthropic/ entry
    so costs are identical regardless of prefix.
    """
    import os
    import litellm

    model = os.environ.get("LLM_MODEL", "")
    # openai/claude-sonnet-4-5-20250929 → anthropic/claude-sonnet-4-5-20250929
    anthropic_key = "anthropic/" + model.split("/", 1)[1]
    base = litellm.get_model_info(anthropic_key)
    pricing = {k: v for k, v in base.items() if v is not None and k != "key"}
    pricing["litellm_provider"] = "openai"
    litellm.register_model({model: pricing})


def apply_all() -> None:
    """Apply every SDK patch. Safe to call more than once."""
    _patch_action_type_factories()
    _patch_terminal_timeout_cap()
    _set_tool_text_content_limit()
    _patch_stream_usage()
    if _is_portkey_openai_claude():
        _patch_preserve_cache_control()
        _register_portkey_anthropic_pricing()
