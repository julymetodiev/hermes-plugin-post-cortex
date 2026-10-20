"""Test harness: a fake post-cortex daemon and the plugin loaded as a package.

Outside a Hermes checkout the three Hermes modules the plugin imports are stubbed
with the same public signatures.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import threading
import types
import uuid
from abc import ABC
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _install_hermes_stubs() -> None:
    try:
        import agent.memory_provider  # noqa: F401
        return
    except ImportError:
        pass
    agent = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")

    @dataclass(frozen=True)
    class RecallStatus:
        provider_label: str
        count: int
        glyph: str = "🧠"

    class MemoryProvider(ABC):
        pass

    def is_trivial_prompt(text):
        s = (text or "").strip()
        return not s or s.startswith("/") or bool(re.match(r"^(ok|thanks|hi|yes|no)[\s!.]*$", s, re.I))

    mp.RecallStatus, mp.MemoryProvider, mp.is_trivial_prompt = RecallStatus, MemoryProvider, is_trivial_prompt
    ss = types.ModuleType("agent.secret_scope")
    ss.get_secret = lambda name, default=None: os.environ.get(name, default)
    hc = types.ModuleType("hermes_constants")
    hc.get_hermes_home = lambda: Path(os.environ.get("HERMES_HOME", "/tmp/hermes-test-home"))
    agent.memory_provider, agent.secret_scope = mp, ss
    sys.modules.update({"agent": agent, "agent.memory_provider": mp, "agent.secret_scope": ss, "hermes_constants": hc})


_install_hermes_stubs()


def _load_plugin():
    spec = importlib.util.spec_from_file_location("post_cortex_plugin", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["post_cortex_plugin"] = mod
    spec.loader.exec_module(mod)
    return mod


plugin = _load_plugin()


class FakeDaemon:
    """Just enough of post-cortex 0.4.0: REST sessions/workspaces, MCP tools over SSE."""

    def __init__(self, token: str = "", allowed_hosts=("127.0.0.1", "localhost"), version: str = "0.4.0"):
        self.token, self.allowed_hosts, self.version = token, set(allowed_hosts), version
        self.workspaces: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self.records: list[dict] = []
        self.search_results: list[dict] = []
        self.calls: list[tuple[str, dict]] = []
        self.mcp_sessions: set[str] = set()
        self.priming = True
        self.messages: list[dict] = []
        self.turn_ids: set[tuple[str, str]] = set()
        self.cards: dict = {"user": None, "workspace": None}
        self.cards_api = True
        daemon = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="application/json", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _guard(self) -> bool:
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
                if host not in daemon.allowed_hosts:
                    self._send(403, b"Forbidden: Host header is not allowed", "text/plain")
                    return False
                if daemon.token and self.headers.get("Authorization") != f"Bearer {daemon.token}":
                    self._send(401, b"Unauthorized", "text/plain")
                    return False
                return True

            def _json(self, obj, code=200):
                self._send(code, json.dumps(obj).encode())

            def do_GET(self):
                if not self._guard():
                    return
                if self.path == "/api/workspaces":
                    return self._json([{"id": w["id"], "name": w["name"]} for w in daemon.workspaces.values()])
                if self.path == "/api/sessions":
                    return self._json([{"id": s["id"], "name": s["name"], "workspace": s.get("workspace")}
                                       for s in daemon.sessions.values()])
                if self.path.startswith("/api/cards") and daemon.cards_api:
                    return self._json(daemon.cards)
                self._send(404)

            def do_POST(self):
                if not self._guard():
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/workspaces":
                    wid = str(uuid.uuid4())
                    daemon.workspaces[wid] = {"id": wid, "name": body["name"]}
                    return self._json({"id": wid})
                if self.path == "/api/sessions":
                    sid = str(uuid.uuid4())
                    daemon.sessions[sid] = {"id": sid, "name": body["name"], "workspace": None}
                    return self._json({"id": sid})
                m = re.fullmatch(r"/api/workspaces/([^/]+)/sessions/([^/]+)", self.path)
                if m:
                    daemon.sessions[m.group(2)]["workspace"] = daemon.workspaces[m.group(1)]["name"]
                    return self._json({"ok": True})
                if self.path == "/mcp":
                    return self._mcp(body)
                if self.path == "/api/messages":
                    appended, skipped = [], 0
                    for m in body["messages"]:
                        key = (body["session_id"], m["turn_id"])
                        if key in daemon.turn_ids:
                            skipped += 1
                            continue
                        daemon.turn_ids.add(key)
                        daemon.messages.append({"session_id": body["session_id"], **m})
                        appended.append(len(daemon.messages) - 1)
                    return self._json({"appended": appended, "skipped": skipped, "vectorized": len(appended)})
                self._send(404)

            def _mcp(self, msg):
                sid = self.headers.get("Mcp-Session-Id")
                method = msg.get("method")
                if method == "initialize":
                    sid = str(uuid.uuid4())
                    daemon.mcp_sessions.add(sid)
                    result = {"protocolVersion": "2025-03-26", "capabilities": {},
                              "serverInfo": {"name": "post-cortex", "version": daemon.version}}
                elif sid not in daemon.mcp_sessions:
                    return self._send(404, b"session not found", "text/plain")
                elif method == "notifications/initialized":
                    return self._send(202)
                elif method == "tools/call":
                    result = daemon.tool(msg["params"]["name"], msg["params"]["arguments"])
                else:
                    result = {}
                payload = json.dumps({"jsonrpc": "2.0", "id": msg.get("id"), "result": result})
                prefix = "id: 0\ndata:\n\n" if daemon.priming else ""
                self._send(200, (prefix + f"data: {payload}\n\n").encode(), "text/event-stream",
                           {"mcp-session-id": sid})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tool(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))

        def text(summary, data=None):
            out = summary + (f"\n\nResults:\n{json.dumps(data)}" if data is not None else "")
            return {"content": [{"type": "text", "text": out}]}

        if name == "semantic_search":
            return text(f"Found {len(self.search_results)} results", {"results": self.search_results})
        if name == "update_conversation_context":
            if not args.get("entities") or not args.get("relations"):
                return {"isError": True, "content": [{"type": "text", "text": "entities and relations are required"}]}
            self.records.append(args)
            return text("Context updated successfully")
        if name == "assemble_context":
            return text("Assembled 1 items", {"formatted_text": f"ctx for {args['query']}", "items": []})
        if name == "admin":
            return text("healthy", {"status": "ok"})
        return {"isError": True, "content": [{"type": "text", "text": f"unknown tool {name}"}]}

    def close(self):
        self.server.shutdown()


@pytest.fixture
def daemon():
    d = FakeDaemon(token="t0k")
    yield d
    d.close()


@pytest.fixture
def home(tmp_path, daemon, monkeypatch):
    h = tmp_path / "profiles" / "eq-sentinel"
    h.mkdir(parents=True)
    (h / "post-cortex.json").write_text(json.dumps({"url": daemon.url, "workspaces": {"eq-*": "Equal", "*": "hermes"}}))
    # Single-profile Hermes resolves secrets from the process environment.
    monkeypatch.setenv("PCX_TOKEN", daemon.token)
    monkeypatch.delenv("PCX_URL", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def provider(home):
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel", agent_context="primary")
    return p
