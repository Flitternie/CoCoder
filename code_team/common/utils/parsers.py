from __future__ import annotations

import json
import re

from json_repair import repair_json

from common.models import JudgeResult


class CheckResultSchemaError(ValueError):
    """Raised when check-task output does not match strict schema."""


def _to_text(raw: str | object) -> str:
    if hasattr(raw, "content"):
        return str(getattr(raw, "content"))
    if hasattr(raw, "to_string"):
        return str(raw.to_string())
    return str(raw)


def parse_json_any(raw: str | object) -> dict | list:
    text = _to_text(raw)

    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates.extend(fenced)

    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        candidates.append(m.group(1))

    candidates.append(text)

    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(c)
                return json.loads(repaired)
            except Exception:
                continue
    return []


def parse_judge_score(raw: str | object) -> JudgeResult:
    text = _to_text(raw)
    score_match = re.search(r"Final Score:\s*\**(\d+)\**", text, flags=re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 0

    keys = [
        "Requirement Coverage",
        "Consistency with Provided Information",
        "Interface Consistency",
        "Dependency Relations",
        "Directory Structure Matching",
        "Interface & Call Relationship Matching",
    ]
    # Build a lookahead pattern that stops capture at any next section header or Final Score
    all_headers = [re.escape(k) for k in keys] + [r"Final\s+Score"]
    stop_pattern = "|".join(all_headers)
    feedback: dict[str, str] = {}
    for key in keys:
        # Use [\s\S]*? (non-greedy, matches newlines) to capture multi-line feedback
        m = re.search(
            rf"{re.escape(key)}:\s*([\s\S]*?)(?=(?:{stop_pattern})\s*:|$)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            feedback[key] = m.group(1).strip()
    return JudgeResult(final_score=score, feedback=feedback)


def parse_file_list(raw: str | object) -> list[str]:
    data = parse_json_any(raw)
    if isinstance(data, list):
        out: list[str] = []
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def validate_test_command_result(data: object) -> tuple[bool, int, int, str]:
    """Validate strict schema for test-command results.

    Expected schema:
    {
      "passed": bool,
      "passed_count": int,
      "total": int,
      "raw_output": str
    }
    """
    if not isinstance(data, dict):
        raise CheckResultSchemaError("result must be a JSON object")

    required = ("passed", "passed_count", "total", "raw_output")
    missing = [k for k in required if k not in data]
    if missing:
        raise CheckResultSchemaError(f"missing required field(s): {', '.join(missing)}")

    passed = data["passed"]
    passed_count = data["passed_count"]
    total = data["total"]
    raw_output = data["raw_output"]

    if type(passed) is not bool:
        raise CheckResultSchemaError("field 'passed' must be bool")
    if isinstance(passed_count, bool) or not isinstance(passed_count, int):
        raise CheckResultSchemaError("field 'passed_count' must be int")
    if isinstance(total, bool) or not isinstance(total, int):
        raise CheckResultSchemaError("field 'total' must be int")
    if not isinstance(raw_output, str):
        raise CheckResultSchemaError("field 'raw_output' must be str")
    if passed_count < 0 or total < 0:
        raise CheckResultSchemaError("field 'passed_count' and 'total' must be >= 0")

    return passed, passed_count, total, raw_output
