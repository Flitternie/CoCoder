"""ClaudeCodeRuntime: runs `claude --print` as a subprocess with project subagents.

The runtime:
1. Copies subagent .md files into workspace/.claude/agents/
2. Writes CLAUDE.md (workflow instructions) to workspace root
3. Invokes `claude --print` with stream-json output for logging
4. Returns the final text response
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from claude_code_agent_team.config import CLAUDE_SUBPROCESS_TIMEOUT
from claude_code_agent_team.system_prompt import build_claude_md

# Path to the bundled subagent .md files
_AGENTS_SRC = Path(__file__).parent / "agents"


class ClaudeCodeRuntime:
    """Runs a single `claude --print` session in the workspace directory.

    Args:
        workspace_dir: Project workspace (claude's working directory).
        run_dir: Run output directory for logs.
        model: Claude model alias (sonnet, opus, haiku) or full model ID.
        timeout: Subprocess timeout in seconds.
        integrity_content: Optional implementation integrity doc content.
    """

    def __init__(
        self,
        workspace_dir: Path,
        run_dir: Path,
        model: str = "sonnet",
        timeout: int = CLAUDE_SUBPROCESS_TIMEOUT,
        integrity_content: str | None = None,
    ):
        self.workspace_dir = workspace_dir.resolve()
        self.run_dir = run_dir
        self.model = model
        self.timeout = timeout
        self.integrity_content = integrity_content

    def _install_subagents(self) -> None:
        """Copy subagent .md files to workspace/.claude/agents/."""
        target = self.workspace_dir / ".claude" / "agents"
        target.mkdir(parents=True, exist_ok=True)
        for agent_file in _AGENTS_SRC.glob("*.md"):
            shutil.copy2(agent_file, target / agent_file.name)

    def _write_claude_md(self) -> None:
        """Write CLAUDE.md with workflow instructions to the workspace root."""
        (self.workspace_dir / "CLAUDE.md").write_text(build_claude_md(self.integrity_content), encoding="utf-8")

    def run(self, initial_prompt: str) -> str:
        """Execute `claude --print` in the workspace and return the final response.

        Args:
            initial_prompt: The task description / initial user message.

        Returns:
            Final text response from claude.
        """
        self._install_subagents()
        self._write_claude_md()

        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stream_log = logs_dir / "stream.jsonl"

        cmd = [
            "claude",
            "--print",
            "--verbose",
            "--output-format", "stream-json",
            "--model", self.model,
            "--dangerously-skip-permissions",
            *[arg for tool in [
                "WebSearch", "WebFetch",
                "Bash(vim *)", "Bash(vi *)", "Bash(nano *)", "Bash(emacs *)",
                "Bash(less *)", "Bash(more *)",
                "Bash(ssh *)", "Bash(ftp *)", "Bash(sftp *)",
                "Bash(mysql *)", "Bash(psql *)", "Bash(sqlite3 *)",
                "Bash(top *)", "Bash(htop *)",
                "Bash(pip install *)", "Bash(pip3 install *)",
                "Bash(apt-get *)", "Bash(apt *)", "Bash(brew *)",
            ] for arg in ("--disallowed-tools", tool)],
        ]

        print(f"[claude_code_agent_team] Running: {' '.join(cmd[:6])} ...")
        result = subprocess.run(
            cmd,
            input=initial_prompt,
            cwd=str(self.workspace_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        # Persist raw stream output
        stream_log.write_text(result.stdout, encoding="utf-8")

        if result.returncode != 0 and result.stderr:
            (logs_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
            print(f"[claude_code_agent_team] stderr: {result.stderr[:500]}")

        return self._extract_final_response(result.stdout)

    @staticmethod
    def _extract_final_response(stream_json_output: str) -> str:
        """Parse stream-json output and return the final result text."""
        final = ""
        for line in stream_json_output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # stream-json emits a "result" event at the end with the final response
                if event.get("type") == "result":
                    final = event.get("result", "")
                # Also capture assistant messages as fallback
                elif event.get("type") == "assistant":
                    content = event.get("message", {}).get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final = block.get("text", "")
                    elif isinstance(content, str):
                        final = content
            except json.JSONDecodeError:
                pass
        return final
