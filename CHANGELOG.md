# Changelog

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
