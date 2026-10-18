"""post-cortex memory provider for Hermes Agent.

post-cortex (https://github.com/julymetodiev/post-cortex) is a self-hosted memory
daemon: typed records (decisions, fixed problems, code changes, ...) linked into an
entity graph and embedded for semantic search. This provider connects a Hermes
profile to it:

- One post-cortex session per profile (default name "hermes:<profile>") inside a
  workspace chosen by profile-name glob, so related profiles share recall.
- prefetch: a workspace semantic search before each turn; weak matches, entity
  descriptions and noise records are dropped, the rest is injected as context.
- on_memory_write: what the agent saves to its built-in memory is mirrored as a
  record (primary agent only).
- Tools: post_cortex_search, post_cortex_record, post_cortex_context.
- No sync_turn: whole turns are noise. Durable knowledge is recorded deliberately.

Settings: $HERMES_HOME/post-cortex.json. Secrets: PCX_TOKEN (and optionally
PCX_URL) through the Hermes secret scope.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt
from agent.secret_scope import get_secret

from .pcx_client import DEFAULT_URL, PcxClient, PcxError

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.0"
CONFIG_FILE = "post-cortex.json"

INTERACTION_TYPES = ("qa", "decision_made", "problem_solved", "code_change", "requirement_added", "concept_defined")
ENTITY_TYPES = ("technology", "concept", "problem", "solution", "decision", "code_component")
RELATION_TYPES = ("depends_on", "implements", "caused_by", "leads_to", "related_to", "required_by",
                  "conflicts_with", "solves")

DEFAULTS: dict[str, Any] = {
    "url": "",                      # empty: PCX_URL, else http://127.0.0.1:3737
    "workspaces": {"*": "hermes"},  # profile-name glob -> workspace name, first match wins
    "session_name": "hermes:{profile}",
    "auto_recall": True,
    "min_score": 0.62,
    "max_items": 5,
    "item_chars": 500,
    "timeout_s": 3.0,
    "mirror_memory_writes": True,
    "tools": True,
    # Records that carry no knowledge of their own (regexes, matched at the start of the text).
    "skip_patterns": [r"^Modified /", r"^Incremental Update\s*$"],
    # Extra entities for mirrored memory notes: [{"pattern": "<regex>", "entity_type": "<type>"}].
    "entity_patterns": [],
}


def load_config(hermes_home: Optional[str]) -> dict:
    """Defaults overlaid with $hermes_home/post-cortex.json; bad values fall back to defaults."""
    cfg = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    if hermes_home:
        path = Path(hermes_home) / CONFIG_FILE
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("post-cortex: %s is not valid JSON (%s); using defaults", path, e)
            raw = {}
        if isinstance(raw, dict):
            for key, default in DEFAULTS.items():
                value = raw.get(key)
                if value is not None and isinstance(value, type(default)) or (
                        isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool)):
                    cfg[key] = value
    return cfg


def workspace_for(profile: str, mapping: dict) -> str:
    for pattern, name in mapping.items():
        if fnmatch.fnmatch(profile, pattern):
            return str(name)
    return "hermes"


def compile_patterns(cfg: dict) -> tuple[list[re.Pattern], list[tuple[re.Pattern, str]]]:
    skips, entities = [], []
    for p in cfg["skip_patterns"]:
        try:
            skips.append(re.compile(str(p)))
        except re.error as e:
            logger.warning("post-cortex: skip pattern %r ignored: %s", p, e)
    for item in cfg["entity_patterns"]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("entity_type") or "concept")
        try:
            entities.append((re.compile(str(item.get("pattern", ""))), kind if kind in ENTITY_TYPES else "concept"))
        except re.error as e:
            logger.warning("post-cortex: entity pattern %r ignored: %s", item, e)
    return skips, entities


def extract_entities(text: str, patterns: list[tuple[re.Pattern, str]], limit: int = 6) -> list[dict]:
    seen, out = set(), []
    for rx, kind in patterns:
        for m in rx.finditer(text):
            name = m.group(0).strip()
            if name and name not in seen:
                seen.add(name)
                out.append({"name": name, "entity_type": kind})
    return out[:limit]


def _tool_error(message: str) -> str:
    return json.dumps({"error": message[:500]}, ensure_ascii=False)


SEARCH_SCHEMA = {
    "name": "post_cortex_search",
    "description": ("Semantic search over post-cortex long-term memory (records from past work by this and related "
                    "agents). Default scope is this profile's workspace. Treat hits as leads and verify them; "
                    "combined_score 0.62+ is related, 0.7+ is close."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A question or the key terms of what you need."},
            "scope": {"type": "string", "enum": ["workspace", "session", "global"],
                      "description": "workspace (default), session (this profile's own records) or global."},
            "limit": {"type": "integer", "description": "Results to return, 1-20 (default 8)."},
            "recency_bias": {"type": "number", "description": "0 = pure relevance (default); 0.5-1.0 favours recent records."},
            "interaction_type": {"type": "array", "items": {"type": "string", "enum": list(INTERACTION_TYPES)},
                                 "description": "Only these record types (session and global scope)."},
            "date_from": {"type": "string", "description": "YYYY-MM-DD lower bound."},
            "date_to": {"type": "string", "description": "YYYY-MM-DD upper bound."},
        },
        "required": ["query"],
    },
}

RECORD_SCHEMA = {
    "name": "post_cortex_record",
    "description": ("Record durable knowledge in post-cortex: a decision and why, a root cause with evidence, a fix "
                    "that worked or did not, a requirement, a definition. One finding per record, written when it is "
                    "confirmed. Not for routine status, raw logs or secrets."),
    "parameters": {
        "type": "object",
        "properties": {
            "interaction_type": {"type": "string", "enum": list(INTERACTION_TYPES)},
            "title": {"type": "string", "description": "One line; searchable."},
            "description": {"type": "string", "description": "Self-contained explanation; the reader will not have this conversation."},
            "details": {"type": "object", "additionalProperties": {"type": "string"},
                        "description": "Optional extra string fields, e.g. evidence, action, refs."},
            "entities": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["name", "entity_type"],
                "properties": {"name": {"type": "string"}, "entity_type": {"type": "string", "enum": list(ENTITY_TYPES)}}},
                "description": "Things the record is about, named the way the system names them. At least one."},
            "relations": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["from_entity", "to_entity", "relation_type", "context"],
                "properties": {"from_entity": {"type": "string"}, "to_entity": {"type": "string"},
                               "relation_type": {"type": "string", "enum": list(RELATION_TYPES)},
                               "context": {"type": "string", "description": "Why the relation holds."}}},
                "description": "At least one; both ends must be listed in entities; no self-relations."},
            "code_reference": {"type": "string", "description": "Optional 'path', 'path:line' or 'path:start-end'."},
        },
        "required": ["interaction_type", "title", "description", "entities", "relations"],
    },
}

CONTEXT_SCHEMA = {
    "name": "post_cortex_context",
    "description": ("What post-cortex knows around named components: matches entity names in the query (no "
                    "embeddings), adds their graph neighbours and reverse dependencies (what breaks if X changes), "
                    "and returns a prompt-ready block. Name the components exactly."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Names the components you care about."},
            "token_budget": {"type": "integer", "description": "Size of the returned block, 500-8000 (default 2000)."},
        },
        "required": ["query"],
    },
}


class PostCortexMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._cfg = load_config(None)
        self._skips, self._entity_patterns = compile_patterns(self._cfg)
        self._client: Optional[PcxClient] = None
        self._profile = "default"
        self._workspace = ""
        self._workspace_id = ""
        self._session_id = ""
        self._session_name = ""
        self._session_names: dict[str, str] = {}
        self._writes = True
        self._last_count: Optional[int] = None
        self._error = ""

    @property
    def name(self) -> str:
        return "post-cortex"

    # ---- availability and setup ----------------------------------------------------------

    def is_available(self) -> bool:
        # No network here. A loopback daemon without auth needs no token, so the provider is
        # available whenever it is selected; initialize() reports an unreachable daemon.
        return True

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {"key": "url", "description": "post-cortex daemon URL", "default": DEFAULT_URL},
            {"key": "token", "description": "post-cortex auth_token (leave empty for a loopback daemon without auth)",
             "secret": True, "required": False, "env_var": "PCX_TOKEN"},
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / CONFIG_FILE
        try:
            current = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            current = {}
        if not isinstance(current, dict):
            current = {}
        current.update({k: v for k, v in (values or {}).items() if k in DEFAULTS and k != "token"})
        try:
            from utils import atomic_json_write
            atomic_json_write(path, current, mode=0o600, sort_keys=True)
        except ImportError:  # outside Hermes (tests)
            path.write_text(json.dumps(current, indent=2, sort_keys=True))

    def get_status_config(self, provider_config: dict) -> dict:
        from hermes_constants import get_hermes_home
        cfg = load_config(str(get_hermes_home()))
        client = self._make_client(cfg)
        try:
            client.call_tool("admin", {"action": "health"}, timeout=float(cfg["timeout_s"]))
            return {"summary": f"post-cortex at {client.url}: reachable"}
        except (PcxError, OSError) as e:
            return {"summary": f"post-cortex at {client.url}: {e}"}

    # ---- lifecycle -----------------------------------------------------------------------

    def _make_client(self, cfg: dict) -> PcxClient:
        url = str(cfg["url"] or "").strip() or (get_secret("PCX_URL", "") or "").strip() or DEFAULT_URL
        token = get_secret("PCX_TOKEN", "") or ""
        return PcxClient(url, token, timeout=float(cfg["timeout_s"]), client_version=PLUGIN_VERSION)

    def initialize(self, session_id: str, **kwargs) -> None:
        home = str(kwargs.get("hermes_home") or "")
        if not home:
            from hermes_constants import get_hermes_home
            home = str(get_hermes_home())
        self._cfg = load_config(home)
        self._skips, self._entity_patterns = compile_patterns(self._cfg)
        home_path = Path(home)
        self._profile = str(kwargs.get("agent_identity") or
                            (home_path.name if home_path.parent.name == "profiles" else "default"))
        self._writes = kwargs.get("agent_context", "primary") == "primary"
        self._workspace = workspace_for(self._profile, self._cfg["workspaces"])
        self._session_name = str(self._cfg["session_name"]).replace("{profile}", self._profile)
        self._error = ""
        client = self._make_client(self._cfg)
        try:
            self._workspace_id, self._session_id = client.ensure_session(
                self._workspace, self._session_name,
                f"Hermes profile {self._profile}: records written by the agent and mirrored built-in memory")
            self._session_names = {(s.get("id") or s.get("session_id")): s.get("name") or "" for s in client.sessions()}
            self._client = client
        except (PcxError, OSError, ValueError) as e:
            self._error = str(e)
            self._client = None
            logger.warning("post-cortex: daemon unreachable at %s (%s); memory disabled for this session", client.url, e)

    def shutdown(self) -> None:
        self._client = None

    # ---- prompt and recall ---------------------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        lines = [f"## Long-term memory: post-cortex (workspace \"{self._workspace}\")",
                 "Related records from past work may be injected before your turns; treat them as leads and "
                 "verify against the live system before acting."]
        if self._cfg["tools"]:
            lines += ["- `post_cortex_search` to look further; `post_cortex_context` for what depends on a named component.",
                      "- `post_cortex_record` for durable knowledge: decisions and why, root causes with evidence, "
                      "fixes that worked or did not. Not routine status, raw logs or secrets.",
                      "- Before your first record, read the format: `skill_view(\"post-cortex:memory\")`."]
        return "\n".join(lines)

    def _format(self, results: list[dict]) -> tuple[str, int]:
        lines = []
        for r in results:
            text = (r.get("text_content") or "").strip()
            kind = r.get("content_type") or ""
            score = r.get("combined_score", r.get("similarity_score", 0.0)) or 0.0
            if kind == "EntityDescription" or score < float(self._cfg["min_score"]) or not text:
                continue
            if any(rx.search(text) for rx in self._skips):
                continue
            limit = int(self._cfg["item_chars"])
            if len(text) > limit:
                text = text[:limit].rstrip() + "…"
            where = self._session_names.get(r.get("session_id", ""), "") or "?"
            lines.append(f"- ({score:.2f}, {kind}, {where}) {text}")
            if len(lines) >= int(self._cfg["max_items"]):
                break
        if not lines:
            return "", 0
        head = f"<post-cortex: {len(lines)} related records from workspace \"{self._workspace}\">"
        return "\n".join([head, *lines, "</post-cortex>"]), len(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_count = None
        if not self._client or not self._cfg["auto_recall"] or is_trivial_prompt(query):
            return ""
        try:
            results = self._client.search(query[:2000], scope="workspace", scope_id=self._workspace_id, limit=12)
        except (PcxError, OSError, ValueError) as e:
            logger.debug("post-cortex prefetch failed: %s", e)
            return ""
        block, count = self._format(results)
        self._last_count = count
        return block

    def recall_status(self) -> Optional[RecallStatus]:
        if not self._last_count:
            return None
        return RecallStatus(provider_label="post-cortex", count=self._last_count)

    # ---- writes --------------------------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        if (not self._client or not self._writes or not self._cfg["mirror_memory_writes"]
                or action not in ("add", "replace") or not (content or "").strip()):
            return
        me = self._session_name
        found = [e for e in extract_entities(content, self._entity_patterns) if e["name"] != me]
        topic = found or [{"name": f"{self._profile} {target} notes", "entity_type": "concept"}]
        entities = [{"name": me, "entity_type": "concept"}, *topic]
        relations = [{"from_entity": me, "to_entity": e["name"], "relation_type": "related_to",
                      "context": f"noted in the built-in {target} memory of profile {self._profile}"} for e in topic]
        text = content.strip()
        try:
            self._client.write(self._session_id, "concept_defined",
                               {"title": text.splitlines()[0][:120], "description": text[:4000],
                                "source": f"hermes built-in {target} memory ({action})"},
                               entities, relations)
        except (PcxError, OSError, ValueError) as e:
            logger.warning("post-cortex: mirroring a %s memory write failed: %s", target, e)

    # ---- tools ---------------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [SEARCH_SCHEMA, RECORD_SCHEMA, CONTEXT_SCHEMA] if self._cfg["tools"] else []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._client:
            return _tool_error(f"post-cortex is not connected: {self._error or 'not initialized'}")
        try:
            if tool_name == "post_cortex_search":
                return self._tool_search(args)
            if tool_name == "post_cortex_record":
                return self._tool_record(args)
            if tool_name == "post_cortex_context":
                return self._tool_context(args)
        except (PcxError, OSError, ValueError) as e:
            return _tool_error(f"{tool_name} failed: {e}")
        return _tool_error(f"unknown tool {tool_name}")

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return _tool_error("query is required")
        scope = args.get("scope") or "workspace"
        if scope not in ("workspace", "session", "global"):
            return _tool_error("scope must be workspace, session or global")
        scope_id = {"workspace": self._workspace_id, "session": self._session_id}.get(scope, "")
        limit = max(1, min(20, int(args.get("limit") or 8)))
        types = [t for t in (args.get("interaction_type") or []) if t in INTERACTION_TYPES]
        results = self._client.search(query, scope=scope, scope_id=scope_id, limit=limit,
                                      recency_bias=args.get("recency_bias"), interaction_type=types or None,
                                      date_from=args.get("date_from"), date_to=args.get("date_to"))
        out = [{"score": round(float(r.get("combined_score") or 0.0), 3), "type": r.get("content_type"),
                "session": self._session_names.get(r.get("session_id", ""), r.get("session_id")),
                "timestamp": r.get("timestamp"), "content_id": r.get("content_id"), "text": r.get("text_content")}
               for r in results]
        return json.dumps({"scope": scope, "results": out}, ensure_ascii=False)

    def _tool_record(self, args: dict) -> str:
        itype = args.get("interaction_type")
        if itype not in INTERACTION_TYPES:
            return _tool_error(f"interaction_type must be one of {', '.join(INTERACTION_TYPES)}")
        title, description = str(args.get("title") or "").strip(), str(args.get("description") or "").strip()
        if not title or not description:
            return _tool_error("title and description are required")
        entities = [{"name": str(e.get("name", "")).strip(), "entity_type": e.get("entity_type")}
                    for e in args.get("entities") or [] if isinstance(e, dict)]
        relations = [r for r in args.get("relations") or [] if isinstance(r, dict)]
        if not entities or not relations:
            return _tool_error("at least one entity and one relation are required")
        content = {"title": title, "description": description}
        for k, v in (args.get("details") or {}).items():
            if k not in content:
                content[str(k)] = str(v)
        self._client.write(self._session_id, itype, content, entities, relations,
                           code_reference=args.get("code_reference") or None)
        return json.dumps({"stored": True, "session": self._session_name}, ensure_ascii=False)

    def _tool_context(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return _tool_error("query is required")
        budget = max(500, min(8000, int(args.get("token_budget") or 2000)))
        text = self._client.assemble(query, workspace_id=self._workspace_id, token_budget=budget)
        return json.dumps({"context": text or "(nothing on record for the named components)"}, ensure_ascii=False)


def register(ctx) -> None:
    ctx.register_memory_provider(PostCortexMemoryProvider())
    skill = Path(__file__).parent / "skills" / "memory" / "SKILL.md"
    if skill.exists() and hasattr(ctx, "register_skill"):
        ctx.register_skill("memory", skill)
