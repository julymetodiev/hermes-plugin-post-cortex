from __future__ import annotations

import json

from conftest import plugin

from post_cortex_plugin.pcx_client import PcxClient, PcxError


def hit(text, score=0.7, kind="ProblemSolution", session_id=""):
    return {"text_content": text, "combined_score": score, "content_type": kind, "session_id": session_id,
            "content_id": "c1", "timestamp": "2026-10-18T10:00:00Z"}


def test_initialize_creates_workspace_and_session_once(daemon, home):
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel")
    assert p._workspace == "Equal" and p._session_name == "hermes:eq-sentinel"
    assert [w["name"] for w in daemon.workspaces.values()] == ["Equal"]
    q = plugin.PostCortexMemoryProvider()
    q.initialize("s2", hermes_home=str(home), agent_identity="eq-sentinel")
    assert q._session_id == p._session_id and len(daemon.sessions) == 1


def test_same_session_name_in_another_workspace_is_not_reused(daemon, home):
    other = PcxClient(daemon.url, daemon.token)
    _, foreign = other.ensure_session("Elsewhere", "hermes:eq-sentinel", "x")
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel")
    assert p._session_id != foreign


def test_prefetch_filters_and_caps(provider, daemon):
    daemon.search_results = [
        hit("svc-social p95 during the sweep", 0.74, session_id=provider._session_id),
        hit("svc-social. Type: CodeComponent", 0.9, kind="EntityDescription"),
        hit("unrelated architecture notes", 0.55),
        hit("Modified /src/main.rs", 0.8),
        *[hit(f"record {i}", 0.7) for i in range(10)],
    ]
    block = provider.prefetch("EqualGrpcSlow on svc-social")
    lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert len(lines) == 5
    assert "hermes:eq-sentinel" in lines[0] and "EntityDescription" not in block and "Modified /" not in block
    assert provider.recall_status().count == 5
    assert provider.prefetch("thanks!") == ""


def test_long_records_are_trimmed(provider, daemon):
    daemon.search_results = [hit("x" * 2000)]
    line = provider.prefetch("a real question about x").splitlines()[1]
    assert line.endswith("…") and len(line) < 600


def test_sse_without_priming_event_also_works(provider, daemon):
    daemon.priming = False
    daemon.search_results = [hit("svc-social record")]
    assert "svc-social record" in provider.prefetch("svc-social latency")


def test_wrong_token_disables_with_a_hint(daemon, home, monkeypatch):
    monkeypatch.setenv("PCX_TOKEN", "wrong")
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel")
    assert p._client is None and "PCX_TOKEN" in p._error
    assert p.system_prompt_block() == "" and p.prefetch("anything at all") == ""
    assert "not connected" in json.loads(p.handle_tool_call("post_cortex_search", {"query": "x"}))["error"]


def test_disallowed_host_error_names_allowed_hosts(daemon):
    client = PcxClient(daemon.url.replace("127.0.0.1", "localhost"), daemon.token)
    daemon.allowed_hosts = {"127.0.0.1"}
    try:
        client.sessions()
        raise AssertionError("expected 403")
    except PcxError as e:
        assert e.status == 403 and "allowed_hosts" in str(e)


def test_stale_mcp_session_is_renewed(provider, daemon):
    daemon.search_results = [hit("first")]
    assert "first" in provider.prefetch("question one")
    daemon.mcp_sessions.clear()  # daemon restarted
    daemon.search_results = [hit("second")]
    assert "second" in provider.prefetch("question two")


def test_memory_write_is_mirrored_with_entities(daemon, home):
    (home / "post-cortex.json").write_text(json.dumps({
        "url": daemon.url, "entity_patterns": [{"pattern": r"\bsvc-[a-z]+\b", "entity_type": "code_component"}]}))
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel")
    p.on_memory_write("add", "memory", "svc-social restarts are safe at night\nmore detail")
    rec = daemon.records[-1]
    assert rec["interaction_type"] == "concept_defined"
    assert rec["content"]["title"] == "svc-social restarts are safe at night"
    assert {"name": "svc-social", "entity_type": "code_component"} in rec["entities"]
    assert rec["relations"][0]["from_entity"] == "hermes:eq-sentinel"
    p.on_memory_write("remove", "memory", "gone")
    assert len(daemon.records) == 1


def test_subagent_does_not_mirror(daemon, home):
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel", agent_context="subagent")
    p.on_memory_write("add", "memory", "something")
    assert daemon.records == []


def test_search_tool_maps_scope_and_filters(provider, daemon):
    daemon.search_results = [hit("svc-social record", session_id=provider._session_id)]
    out = json.loads(provider.handle_tool_call("post_cortex_search", {
        "query": "svc-social", "scope": "session", "limit": 50, "interaction_type": ["problem_solved", "bogus"]}))
    name, args = daemon.calls[-1]
    assert name == "semantic_search" and args["scope_id"] == provider._session_id and args["limit"] == 20
    assert args["interaction_type"] == ["problem_solved"]
    assert out["results"][0]["session"] == "hermes:eq-sentinel"
    json.loads(provider.handle_tool_call("post_cortex_search", {"query": "x", "scope": "global"}))
    assert "scope_id" not in daemon.calls[-1][1]
    json.loads(provider.handle_tool_call("post_cortex_search", {"query": "x", "corpus": "messages"}))
    assert daemon.calls[-1][1]["corpus"] == "messages"
    bad = json.loads(provider.handle_tool_call("post_cortex_search", {"query": "x", "corpus": "chats"}))
    assert "corpus" in bad["error"]


def test_record_tool_validates_and_writes(provider, daemon):
    bad = json.loads(provider.handle_tool_call("post_cortex_record", {
        "interaction_type": "problem_solved", "title": "t", "description": "d", "entities": [], "relations": []}))
    assert "entity" in bad["error"]
    ok = json.loads(provider.handle_tool_call("post_cortex_record", {
        "interaction_type": "problem_solved", "title": "p95 spike", "description": "caused by the sweep",
        "details": {"evidence": "5.0 s"},
        "entities": [{"name": "svc-social", "entity_type": "code_component"}, {"name": "sweep", "entity_type": "problem"}],
        "relations": [{"from_entity": "sweep", "to_entity": "svc-social", "relation_type": "related_to", "context": "c"}],
        "code_reference": "ops/sweep.sh:10", "derived_from": ["message:4"]}))
    assert ok["stored"] is True
    rec = daemon.records[-1]
    assert rec["session_id"] == provider._session_id and rec["content"]["evidence"] == "5.0 s"
    assert rec["code_reference"] == "ops/sweep.sh:10"
    assert rec["derived_from"] == ["message:4"]


def test_context_tool(provider, daemon):
    out = json.loads(provider.handle_tool_call("post_cortex_context", {"query": "svc-social"}))
    assert out["context"] == "ctx for svc-social"
    assert daemon.calls[-1][1]["workspace_id"] == provider._workspace_id


def test_tools_can_be_turned_off(daemon, home):
    (home / "post-cortex.json").write_text(json.dumps({"url": daemon.url, "tools": False}))
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="x")
    assert p.get_tool_schemas() == [] and p._workspace == "hermes"


def test_bad_config_falls_back_to_defaults(tmp_path):
    (tmp_path / "post-cortex.json").write_text("{not json")
    assert plugin.load_config(str(tmp_path))["min_score"] == 0.62
    (tmp_path / "post-cortex.json").write_text(json.dumps({"min_score": "high", "max_items": 3, "timeout_s": 5}))
    cfg = plugin.load_config(str(tmp_path))
    assert cfg["min_score"] == 0.62 and cfg["max_items"] == 3 and cfg["timeout_s"] == 5


def test_save_config_keeps_token_out_of_the_file(tmp_path):
    plugin.PostCortexMemoryProvider().save_config({"url": "http://pcx:3737", "token": "secret"}, str(tmp_path))
    saved = json.loads((tmp_path / "post-cortex.json").read_text())
    assert saved == {"url": "http://pcx:3737"}



def _messages(*turns):
    """OpenAI-style history: (user text, [(tool name, args)], reply) per turn."""
    out = []
    for user, calls, reply in turns:
        out.append({"role": "user", "content": user})
        if calls:
            out.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]})
            out.append({"role": "tool", "content": "tool output, never sent"})
        out.append({"role": "assistant", "content": reply})
    return out


def test_capture_is_off_by_default(provider, daemon):
    provider.sync_turn("restart svc-social?", "done", session_id="h1", messages=_messages(("q", [], "a")))
    assert daemon.messages == []


def test_turns_are_captured_with_tools_files_and_authors(daemon, home):
    (home / "post-cortex.json").write_text(json.dumps({"url": daemon.url, "capture": True}))
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel", agent_context="primary")
    history = _messages(("first question", [], "first answer"),
                        ("restart svc-social?", [("terminal", {"command": "systemctl restart x"}),
                                                 ("read_file", {"path": "/etc/svc.conf"})], "restarted"))
    p.sync_turn("restart svc-social?", "restarted", session_id="h1", messages=history,
                turn_author={"id": "42", "name": "Julius", "is_bot": False})
    user, reply = daemon.messages
    assert user["turn_id"] == "h1:2:user" and user["author"] == "Julius" and user["role"] == "user"
    assert reply["turn_id"] == "h1:2:assistant" and reply["author"] == "hermes:eq-sentinel"
    assert reply["tools"] == ["terminal", "read_file"] and reply["files"] == ["/etc/svc.conf"]
    assert "tool output" not in json.dumps(daemon.messages)
    assert {m["session_id"] for m in daemon.messages} == {p._session_id}
    # A resent turn is skipped by the daemon (same turn ids).
    p.sync_turn("restart svc-social?", "restarted", session_id="h1", messages=history)
    assert len(daemon.messages) == 2


def test_subagent_does_not_capture(daemon, home):
    (home / "post-cortex.json").write_text(json.dumps({"url": daemon.url, "capture": True}))
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel", agent_context="subagent")
    p.sync_turn("q", "a", session_id="h1")
    assert daemon.messages == []


def test_records_carry_the_profile_as_author(provider, daemon):
    provider.on_memory_write("add", "memory", "svc-social restarts are safe at night")
    assert daemon.records[-1]["author"] == "hermes:eq-sentinel"


def test_cards_go_into_the_system_prompt(daemon, home):
    daemon.cards = {"user": {"version": 1, "text": "Answer briefly."},
                    "workspace": {"version": 3, "text": "## Goals\nship"}}
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel")
    block = p.system_prompt_block()
    assert "### User card\nAnswer briefly." in block
    assert "### Project card (Equal)\n## Goals\nship" in block


def test_daemon_without_cards_api_still_works(daemon, home):
    daemon.cards_api = False
    p = plugin.PostCortexMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_identity="eq-sentinel")
    assert p._client is not None and "card" not in p.system_prompt_block().lower()
