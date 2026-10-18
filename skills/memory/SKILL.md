---
name: memory
description: How to search and record long-term memory in post-cortex (decisions, root causes, fixes) so records join the knowledge graph and other agents can find them.
---

# post-cortex memory

Long-term memory shared by the Hermes profiles of a workspace (and by any other
post-cortex client, such as a coding agent, that uses the same workspace). Related
records are injected before your turns; these tools look further and record what is
worth keeping. Records are leads: confirm them against the live system before acting.

## Search

- `post_cortex_search` — `{"query": "...", "limit": 8}` searches your workspace.
  Write the query as a question or the key terms ("p95 spike during a backfill"), not one word.
  - `scope`: `workspace` (default), `session` (only your own records) or `global`.
  - `recency_bias`: 0.5–1.0 favours recent records ("what just broke"); 0 for timeless ones.
  - `date_from` / `date_to` (`YYYY-MM-DD`) bound the time.
  - `interaction_type` (e.g. `["problem_solved"]`) filters in `session` and `global` scope;
    `workspace` scope ignores it.
  - Scores: unrelated text still scores about 0.55 with the default embedding model;
    0.62+ is related, 0.7+ is close. Skip results that only say "Modified <path>".
- `post_cortex_context` — `{"query": "orders-api billing-worker"}`: no embeddings; it matches
  entity names in the query, adds their graph neighbours and the reverse dependencies
  (what breaks if X changes). Name the components exactly as they are recorded.

## Record

Record when the knowledge will matter later to you or another agent: a decision and why,
a root cause and the evidence, a fix that worked (or did not), a constraint or requirement,
a gotcha that cost time. Not: routine status, raw logs, anything already in the docs.
One record per finding; write it when the finding is confirmed, not speculatively.

`post_cortex_record`:

```json
{
  "interaction_type": "problem_solved",
  "title": "orders-api p95 at 5 s during the nightly backfill",
  "description": "The latency alert on orders-api was the nightly backfill job calling RecomputeTotals at ~1 req/s; no user impact; it cleared when the job ended (74 min).",
  "details": {
    "evidence": "p95 of RecomputeTotals = 5.0 s; 1287 calls, 0 errors",
    "action": "none; move the backfill off peak if it recurs"
  },
  "entities": [
    {"name": "HighLatency", "entity_type": "problem"},
    {"name": "orders-api", "entity_type": "code_component"},
    {"name": "backfill-totals", "entity_type": "code_component"}
  ],
  "relations": [
    {"from_entity": "HighLatency", "to_entity": "backfill-totals", "relation_type": "caused_by",
     "context": "the backfill's RecomputeTotals calls pushed p95 to 5 s"},
    {"from_entity": "HighLatency", "to_entity": "orders-api", "relation_type": "related_to",
     "context": "the alert fired on orders-api"}
  ],
  "code_reference": "jobs/backfill_totals.py:40"
}
```

Rules the server enforces (a violating record is rejected):
- `interaction_type`: `qa`, `decision_made`, `problem_solved`, `code_change`, `requirement_added`, `concept_defined`.
- `title` and `description` carry the searchable text; write them self-contained, the reader
  will not have your conversation. `details` takes extra string fields.
- `entities`: at least one; `entity_type` one of `technology`, `concept`, `problem`, `solution`,
  `decision`, `code_component`.
- `relations`: at least one; `relation_type` one of `depends_on`, `implements`, `caused_by`,
  `leads_to`, `related_to`, `required_by`, `conflicts_with`, `solves`; both ends must be in
  `entities`; no self-relations; `context` says why.
- `code_reference` (optional): `"path"`, `"path:line"` or `"path:start-end"`.

Name entities the way the system names them (service, alert, ticket, migration, host, file
path as in the repo), so records join up in the graph. Reuse an existing name rather than
inventing a variant: search first when unsure.

Secrets and personal data are redacted on the server (`[REDACTED:<kind>]`); still do not
paste tokens, passwords, e-mail addresses or phone numbers.

The reply is `{"stored": true}`; the record is stored when it returns. If a write fails,
say so and carry on with the task; do not retry in a loop.
