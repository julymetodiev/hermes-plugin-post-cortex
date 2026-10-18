"""Minimal post-cortex client (standard library only).

MCP over Streamable HTTP for the search and write tools, the REST API for
sessions and workspaces. The caller passes the URL and the bearer token; this
module never reads credentials on its own.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_URL = "http://127.0.0.1:3737"
PROTOCOL = "2025-03-26"
CLIENT_NAME = "hermes-plugin-post-cortex"


class PcxError(RuntimeError):
    """A request to the daemon failed; the message says which and why."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class PcxClient:
    def __init__(self, url: str = DEFAULT_URL, token: str = "", timeout: float = 10.0, client_version: str = "0"):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.client_version = client_version
        self._mcp_session: Optional[str] = None
        self._lock = threading.Lock()
        self._next_id = 0

    # ---- transport ------------------------------------------------------------------------

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if extra:
            h.update(extra)
        return h

    @staticmethod
    def _http_error(what: str, e: urllib.error.HTTPError) -> PcxError:
        body = e.read()[:200].decode("utf-8", "replace")
        hint = {401: " (wrong or missing PCX_TOKEN)",
                403: " (the daemon's allowed_hosts does not list the host in the URL)"}.get(e.code, "")
        return PcxError(f"{what}: HTTP {e.code}{hint} {body}".strip(), e.code)

    def _rest(self, method: str, path: str, body: Any = None, timeout: Optional[float] = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise self._http_error(f"{method} {path}", e) from e
        return json.loads(raw) if raw else None

    def _post_mcp(self, payload: dict, timeout: Optional[float]) -> Optional[dict]:
        extra = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": PROTOCOL}
        if self._mcp_session:
            extra["Mcp-Session-Id"] = self._mcp_session
        req = urllib.request.Request(self.url + "/mcp", data=json.dumps(payload).encode(),
                                     method="POST", headers=self._headers(extra))
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                sid = r.headers.get("mcp-session-id")
                if sid:
                    self._mcp_session = sid
                ctype = r.headers.get("Content-Type", "")
                raw = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404 and self._mcp_session:  # daemon restarted: the MCP session is gone
                self._mcp_session = None
            raise self._http_error(f"mcp {payload.get('method')}", e) from e
        if "id" not in payload:  # notification
            return None
        if "text/event-stream" in ctype:
            for line in raw.splitlines():
                data = line[5:].strip() if line.startswith("data:") else ""
                if data:  # rmcp >= 3 opens the stream with an empty priming event
                    msg = json.loads(data)
                    if msg.get("id") == payload["id"]:
                        return msg
            raise PcxError(f"mcp {payload.get('method')}: no response in the event stream")
        return json.loads(raw)

    def _rpc(self, method: str, params: dict, timeout: Optional[float] = None) -> dict:
        self._next_id += 1
        msg = self._post_mcp({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}, timeout)
        if msg is None or "error" in msg:
            raise PcxError(f"mcp {method}: {(msg or {}).get('error')}")
        return msg["result"]

    def _ensure_mcp(self, timeout: Optional[float]) -> None:
        if self._mcp_session:
            return
        self._rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                       "clientInfo": {"name": CLIENT_NAME, "version": self.client_version}}, timeout)
        self._post_mcp({"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout)

    def call_tool(self, name: str, arguments: dict, timeout: Optional[float] = None) -> Any:
        """Call an MCP tool; returns the JSON payload of its result (None when it has none)."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    self._ensure_mcp(timeout)
                    res = self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout)
                    break
                except PcxError:
                    # A stale MCP session was dropped by _post_mcp: retry once with a new one.
                    if attempt == 2 or self._mcp_session:
                        raise
        if res.get("isError"):
            text = " ".join(c.get("text", "") for c in res.get("content", []))
            raise PcxError(f"{name}: {text[:300]}")
        if res.get("structuredContent") is not None:
            return res["structuredContent"]
        # The daemon returns a human summary first and the JSON payload after "Results:".
        for c in reversed(res.get("content", [])):
            text = c.get("text", "")
            starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
            if starts:
                try:
                    return json.loads(text[min(starts):])
                except json.JSONDecodeError:
                    continue
        return None

    # ---- REST: sessions / workspaces -----------------------------------------------------

    def workspaces(self) -> list[dict]:
        d = self._rest("GET", "/api/workspaces")
        return d if isinstance(d, list) else (d or {}).get("workspaces", [])

    def sessions(self) -> list[dict]:
        d = self._rest("GET", "/api/sessions")
        return d if isinstance(d, list) else (d or {}).get("sessions", [])

    def create_workspace(self, name: str, description: str) -> str:
        d = self._rest("POST", "/api/workspaces", {"name": name, "description": description})
        return d.get("id") or d.get("workspace_id")

    def create_session(self, name: str, description: str) -> str:
        d = self._rest("POST", "/api/sessions", {"name": name, "description": description})
        return d.get("id") or d.get("session_id")

    def attach(self, workspace_id: str, session_id: str, role: str = "related") -> None:
        self._rest("POST", f"/api/workspaces/{workspace_id}/sessions/{session_id}", {"role": role})

    def ensure_session(self, workspace_name: str, session_name: str, description: str) -> tuple[str, str]:
        """(workspace_id, session_id); creates either when missing. A session counts as the
        same one only when its name matches inside this workspace."""
        ws = [w for w in self.workspaces() if w.get("name") == workspace_name]
        ws_id = (ws[0].get("id") or ws[0].get("workspace_id")) if ws else \
            self.create_workspace(workspace_name, f"{workspace_name} (created by Hermes)")
        same = [s for s in self.sessions() if s.get("name") == session_name and s.get("workspace") == workspace_name]
        if same:
            return ws_id, same[0].get("id") or same[0].get("session_id")
        s_id = self.create_session(session_name, description)
        self.attach(ws_id, s_id)
        return ws_id, s_id

    # ---- MCP tools ------------------------------------------------------------------------

    def search(self, query: str, *, scope: str = "workspace", scope_id: str = "", limit: int = 8,
               timeout: Optional[float] = None, **filters: Any) -> list[dict]:
        args: dict[str, Any] = {"query": query, "scope": scope, "limit": limit}
        if scope != "global":
            args["scope_id"] = scope_id
        args.update({k: v for k, v in filters.items() if v not in (None, "", [])})
        d = self.call_tool("semantic_search", args, timeout)
        return (d or {}).get("results", []) if isinstance(d, dict) else []

    def assemble(self, query: str, *, workspace_id: str, token_budget: int = 2000,
                 timeout: Optional[float] = None) -> str:
        d = self.call_tool("assemble_context", {"query": query, "workspace_id": workspace_id,
                                                "token_budget": token_budget}, timeout)
        return str((d or {}).get("formatted_text") or "") if isinstance(d, dict) else ""

    def write(self, session_id: str, interaction_type: str, content: dict, entities: list[dict],
              relations: list[dict], *, code_reference: Any = None, timeout: Optional[float] = None) -> Any:
        args: dict[str, Any] = {"session_id": session_id, "interaction_type": interaction_type,
                                "content": {str(k): str(v) for k, v in content.items()},
                                "entities": entities, "relations": relations}
        if code_reference:
            args["code_reference"] = code_reference
        return self.call_tool("update_conversation_context", args, timeout)
