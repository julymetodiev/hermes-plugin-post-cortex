# Changelog

## [0.2.0] - 2026-10-20

Needs post-cortex 0.5.0 for the new features; with an older daemon they are skipped.

- Cards: the user card and the workspace's project card are put in the system prompt
  (`cards`, on by default).
- Authorship: records (mirrored memory writes, `post_cortex_record`) and captured messages
  carry the author `hermes:<profile>`; a captured user message carries the sender's display
  name from Hermes when known.
- `post_cortex_search` takes `corpus` (`records`, `messages`, `all`); `post_cortex_record` takes
  `derived_from`, the `message:<n>` ids a record was drawn from. Hits show `author` and
  `derived_from` when set.
- Conversation capture (`capture`, off by default): `sync_turn` sends each completed turn of
  the primary agent to the profile session's message stream: the user's message, the final
  reply, tool names and file paths (never tool output). Turn ids are
  `<hermes session>:<n>:user|assistant`, so a resent turn is stored once.

## [0.1.0] - 2026-10-18

First release.

- Memory provider `post-cortex`: workspace recall before each turn (score threshold, entity
  descriptions and noise records dropped), built-in memory writes mirrored as records
  (primary agent only).
- One post-cortex session per profile inside a workspace chosen by profile-name glob;
  missing workspaces and sessions are created.
- Tools `post_cortex_search`, `post_cortex_record`, `post_cortex_context`.
- Skill `post-cortex:memory`: record format and search tips.
- Connection: `url` / `PCX_URL`, `PCX_TOKEN` through the Hermes secret scope. Error messages
  name the cause of 401 (token) and 403 (daemon `allowed_hosts`).
- Requires post-cortex 0.4.0 or newer (unified search result fields; empty SSE priming
  event of rmcp 3 is skipped).
