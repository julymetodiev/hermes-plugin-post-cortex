# post-cortex memory for Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory provider for
[post-cortex](https://github.com/julymetodiev/post-cortex), a self-hosted memory daemon.
post-cortex stores typed records (decisions, fixed problems, code changes, requirements,
definitions) in an entity graph and embeds them for semantic search. Your data stays on
your own machines: the daemon runs where you put it, with RocksDB or SurrealDB storage and
local embeddings.

What the plugin does:

- **Recall before each turn.** A semantic search over the profile's workspace; weak matches,
  bare entity descriptions and noise records are dropped, and up to five related records are
  injected as context.
- **Shared memory by workspace.** Each profile writes to its own post-cortex session
  (`hermes:<profile>` by default). Profiles are mapped to workspaces by name glob, so a team
  of profiles (and other post-cortex clients, such as a coding agent working on the same
  project) recall each other's records.
- **Built-in memory mirrored.** What the agent saves with the built-in memory tool is also
  stored as a record (primary agent only; cron and subagent runs never write automatically).
- **Tools:** `post_cortex_search`, `post_cortex_record`, `post_cortex_context` (graph
  neighbourhood and reverse dependencies of named components).
- **Skill:** `post-cortex:memory`, the record format and search tips, loaded with
  `skill_view("post-cortex:memory")`.

No conversation turns are captured automatically: whole turns are noise. Agents record
durable knowledge deliberately.

## Requirements

- Hermes Agent 0.22.2 or newer.
- A post-cortex daemon, **0.4.0 or newer**, reachable over HTTP (`pcx start`). No Python
  dependencies beyond the standard library.

## Install

```bash
hermes plugins install post-cortex
hermes memory setup          # pick post-cortex; enter the URL and, if set, the auth token
```

or set it by hand in the profile's `config.yaml`:

```yaml
memory:
  provider: post-cortex
```

Start a new session to activate it. `hermes memory status` shows whether the daemon is reachable.

## Connecting to the daemon

| Setting | Where | Default |
|---|---|---|
| URL | `url` in `post-cortex.json`, else `PCX_URL` | `http://127.0.0.1:3737` |
| Token | `PCX_TOKEN` (profile `.env` / secret scope) | none |

A daemon on loopback without `auth_token` needs neither. For a daemon on another host:

- set `auth_token` in the daemon's `daemon.toml` and the same value as `PCX_TOKEN`;
- list the host name or IP you put in the URL in the daemon's `allowed_hosts`, otherwise every
  request gets `403 Forbidden: Host header is not allowed`;
- use a private network (Tailscale, WireGuard) or TLS in front of the daemon: the plugin speaks
  plain HTTP.

## Configuration

Optional settings live in `$HERMES_HOME/post-cortex.json` (per profile). Everything has a default.

```json
{
  "url": "http://127.0.0.1:3737",
  "workspaces": {"ops-*": "Ops", "dev-*": "MyProject", "*": "hermes"},
  "session_name": "hermes:{profile}",
  "auto_recall": true,
  "min_score": 0.62,
  "max_items": 5,
  "item_chars": 500,
  "timeout_s": 3.0,
  "mirror_memory_writes": true,
  "tools": true,
  "skip_patterns": ["^Modified /", "^Incremental Update\\s*$"],
  "entity_patterns": [
    {"pattern": "\\b[a-z]+-api\\b", "entity_type": "code_component"}
  ]
}
```

| Key | Meaning |
|---|---|
| `workspaces` | Profile-name glob → workspace name; first match wins. Missing workspaces and sessions are created. |
| `session_name` | Session per profile; `{profile}` is replaced with the profile name. |
| `min_score` | Recall threshold on `combined_score`. With the default Potion embedding model unrelated text scores about 0.55; 0.62 keeps related records only. |
| `max_items`, `item_chars` | Size of the injected recall block. |
| `timeout_s` | Per request. Recall is skipped, not retried, when the daemon is slow or down. |
| `tools` | `false` to drop the three tools (for example when the post-cortex MCP server is configured in Hermes directly). |
| `skip_patterns` | Regexes for records that carry no knowledge; they are never injected. |
| `entity_patterns` | Regexes that name entities in mirrored memory notes, so they join the graph. |

## Using the post-cortex MCP server as well

The plugin's tools cover search, recording and dependency tracing. For the full tool set
(structured summaries, entity queries), add the daemon's MCP endpoint to Hermes as well:

```yaml
mcp_servers:
  post-cortex:
    url: http://127.0.0.1:3737/mcp
    headers:
      Authorization: "Bearer ${PCX_TOKEN}"
    tools:
      include: [semantic_search, assemble_context, query_conversation_context, get_structured_summary]
```

## Data and network

- **Where data goes.** Only to the post-cortex daemon at the configured URL. Each non-trivial
  user prompt (up to 2,000 characters) is sent as a search query before the turn. Built-in
  memory writes and `post_cortex_record` calls are sent as records. Nothing is sent anywhere else.
- **What is stored.** The daemon stores records, entities and relations in its own database. The
  daemon redacts credentials, e-mail addresses, phone numbers and public IPv4 addresses on write.
- **Credentials.** `PCX_TOKEN` is read through the Hermes secret scope and never written to
  `post-cortex.json`.
- **State on this machine.** Only `$HERMES_HOME/post-cortex.json`. The plugin starts no
  background threads or processes and never deletes data in post-cortex.

## Development

```bash
uvx --with pytest pytest tests -q          # tests use a fake daemon and stub Hermes modules
hermes plugins validate . --install-deps
```

## License

MIT
