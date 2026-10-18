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


def test_record_tool_validates_and_writes(provider, daemon):
    bad = json.loads(provider.handle_tool_call("post_cortex_record", {
        "interaction_type": "problem_solved", "title": "t", "description": "d", "entities": [], "relations": []}))
    assert "entity" in bad["error"]
    ok = json.loads(provider.handle_tool_call("post_cortex_record", {
        "interaction_type": "problem_solved", "title": "p95 spike", "description": "caused by the sweep",
        "details": {"evidence": "5.0 s"},
        "entities": [{"name": "svc-social", "entity_type": "code_component"}, {"name": "sweep", "entity_type": "problem"}],
        "relations": [{"from_entity": "sweep", "to_entity": "svc-social", "relation_type": "related_to", "context": "c"}],
        "code_reference": "ops/sweep.sh:10"}))
    assert ok["stored"] is True
    rec = daemon.records[-1]
    assert rec["session_id"] == provider._session_id and rec["content"]["evidence"] == "5.0 s"
    assert rec["code_reference"] == "ops/sweep.sh:10"


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

