"""Lightweight local proxy: OpenAI-format requests -> SigV4-signed Bedrock Mantle.

Kimi K2.5 on Bedrock has tool-call parsing bugs via the Converse API.
This proxy lets LiteLLM talk plain OpenAI format to localhost; the proxy
signs each request with AWS SigV4 and forwards it to the Mantle
(bedrock-mantle.<region>.api.aws) Chat Completions endpoint.
"""
from __future__ import annotations

import atexit
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session
import httpx

log = logging.getLogger(__name__)

_TRUNCATED_ARGS = json.dumps({"error": "arguments were truncated by the model"})


class MantleProxy:
    """Local HTTP server that forwards to Bedrock Mantle with SigV4 auth."""

    def __init__(self, region: str):
        self.region = region
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._client = httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10))
        self._boto_session = None  # lazily created, reused across requests

    @property
    def base_url(self) -> str:
        assert self._server is not None, "proxy not started"
        _, port = self._server.server_address[:2]
        return f"http://127.0.0.1:{port}/v1"

    def start(self) -> None:
        if self._server is not None:
            return

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                body = outer._prepare_request_body(body)
                target = f"https://bedrock-mantle.{outer.region}.api.aws{self.path}"

                try:
                    headers = outer._sigv4_headers("POST", target, body)
                    with outer._client.stream("POST", target, headers=headers, content=body) as resp:
                        self.send_response(resp.status_code)
                        skip = {"connection", "keep-alive", "transfer-encoding",
                                "content-encoding", "content-length"}
                        for k, v in resp.headers.items():
                            if k.lower() not in skip:
                                self.send_header(k, v)
                        self.end_headers()
                        for chunk in resp.iter_bytes():
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except BrokenPipeError:
                    return
                except Exception as exc:
                    msg = f"Mantle proxy error: {exc}".encode()
                    self.send_response(502)
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)

            def log_message(self, *_a) -> None:
                pass  # suppress noisy per-request logs

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        srv.daemon_threads = True
        self._server = srv
        self._thread = threading.Thread(target=srv.serve_forever, daemon=True,
                                        name="bedrock-mantle-proxy")
        self._thread.start()

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._client.close()

    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_request_body(body: bytes) -> bytes:
        """Sanitize request body before forwarding to Mantle.

        1. Strip provider prefixes from model field.
        2. Fix truncated tool_call arguments that Kimi K2.5 sometimes produces.
           Mantle rejects requests where arguments is not valid JSON.
        """
        if not body:
            return body
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return body

        # --- strip model prefix ---
        model = data.get("model", "")
        if isinstance(model, str):
            for pfx in ("bedrock/", "openai/", "custom_openai/"):
                while model.startswith(pfx):
                    model = model[len(pfx):]
            data["model"] = model

        # --- fix broken tool_call arguments ---
        for msg in data.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                args = (tc.get("function") or {}).get("arguments")
                if not isinstance(args, str):
                    continue
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    tool_name = (tc.get("function") or {}).get("name", "?")
                    log.warning("Replacing truncated tool_call arguments "
                                "for '%s' (len=%d)", tool_name, len(args))
                    tc["function"]["arguments"] = _TRUNCATED_ARGS

        return json.dumps(data, separators=(",", ":")).encode()

    def _sigv4_headers(self, method: str, url: str, body: bytes) -> dict[str, str]:
        if self._boto_session is None:
            self._boto_session = Session()
        creds = self._boto_session.get_credentials()
        if creds is None:
            raise RuntimeError("AWS credentials not found for Bedrock Mantle")
        headers = {
            "Content-Type": "application/json",
            "Host": f"bedrock-mantle.{self.region}.api.aws",
        }
        req = AWSRequest(method=method, url=url, data=body, headers=headers)
        SigV4Auth(creds, "bedrock", self.region).add_auth(req)
        return dict(req.headers.items())


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------
_proxy: MantleProxy | None = None
_proxy_lock = threading.Lock()


def ensure_mantle_proxy(region: str) -> MantleProxy:
    """Start the singleton Mantle proxy (idempotent, thread-safe)."""
    global _proxy
    if _proxy is not None:
        return _proxy
    with _proxy_lock:
        if _proxy is None:
            p = MantleProxy(region)
            p.start()
            atexit.register(p.close)
            _proxy = p
    return _proxy
