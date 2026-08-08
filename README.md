# NeuramineRL

**Self-improvement for AI agents.** NeuramineRL gives your agent the ability to learn from its
past mistakes

Every time your agent fails, NeuramineRL reflects on the failure and distills it into a
*conditioned lesson* ("When submitting the booking form, use ISO dates; MM/DD/YYYY is silently
rejected"). On future runs, the top-k lessons relevant to *that task* (default 5) are retrieved
and injected into the prompt under a hard token budget (default 800) — the block never grows
with history, no matter how many lessons have accumulated.
Crucially, NeuramineRL then **tracks whether each injected lesson actually improved outcomes** —
lessons that help get promoted, lessons that don't decay and get pruned. No pile of stale
superstitions.

```
run agent → capture trajectory → detect outcome → reflect on failures
     ↑                                                    │
     └── inject lessons ← score & prune ← store lessons ←─┘
```

## Quickstart

```bash
pip install neuraminerl[embeddings]
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY / GEMINI_API_KEY — used for reflection
```

```python
from neuraminerl import Learner

nm = Learner()  # zero config: SQLite + local embeddings in ./.neuraminerl/

with nm.run(task="Book the cheapest NYC->SFO flight on the demo site") as run:
    prompt = SYSTEM_PROMPT + str(run.lessons)  # inject lessons from past failures
    result = my_agent(prompt)  # your agent, unchanged
    run.log(result.messages)  # best-effort trajectory capture
    run.end(success=result.ok, error=result.error)
```

Run it twice. The second run is smarter.

On failure, NeuramineRL reflects (one cheap LLM call, off the hot path) and stores lessons like:

```
<learned_lessons>
Lessons from previous attempts at similar tasks. Apply them unless clearly
inapplicable to the current situation.
1. When submitting the booking form, use ISO dates (YYYY-MM-DD); MM/DD/YYYY is silently rejected.
2. When an API call returns 409, retry once with a new idempotency key instead of changing the payload.
</learned_lessons>
```

## Why not just a memory library?

Storing lessons is the easy part. The hard parts — the parts NeuramineRL owns — are:

1. **Outcome capture** — failures detected from exceptions, explicit results, delayed user
   feedback (`nm.feedback(run_id, "that was wrong", success=False)`), or an optional LLM judge.
2. **Reflection** — failures are distilled into *conditioned* rules ("when X, do Y"), not vague
   advice, and deduplicated/generalized against existing lessons at write time.
3. **Lesson lifecycle** — every injection is recorded; run outcomes feed back into each lesson's
   evidence (a Beta-Bernoulli model with time decay). A lesson is only kept if its
   pessimistic success estimate beats your agent's baseline. Helpful lessons get promoted,
   useless ones retire automatically.
4. **Zero-config, local-first** — SQLite + local static embeddings. Nothing leaves your machine
   except the reflection call. No telemetry.

## Fitting your stack

**Any LLM provider.** Reflection is one small call off the hot path, so it does not have to match
whatever model your agent runs on.

```python
Learner(llm="anthropic:claude-haiku-4-5")
Learner(llm="gemini:gemini-2.5-flash")
Learner(llm="openai:gpt-4o-mini")
Learner(llm="openai:llama-3.1-70b@https://api.groq.com/openai/v1")  # any OpenAI-compatible host
Learner(llm="openai:qwen2.5@http://localhost:11434/v1")  # local Ollama/vLLM, no key
```

The `@base_url` suffix points the `openai` provider at anything speaking Chat Completions — Groq,
Together, Fireworks, DeepSeek, OpenRouter, Azure OpenAI, Ollama, vLLM. Keys are read from
`NEURAMINERL_API_KEY` first, then the provider's usual variable; a custom `base_url` may be
keyless. With no `llm=` argument, the provider is detected from whichever key is present, in the
order Anthropic, OpenAI, Gemini. `NEURAMINERL_LLM` sets the same spec by environment. For
anything else (Bedrock,
Vertex AI, Cohere, an in-house gateway), implement the four-argument `LLMClient` protocol and
pass the instance: `Learner(llm=my_client)`.

**Any database.** SQLite is the zero-config default and assumes one process. For Celery workers,
multiple containers, or anything else where writers do not share a disk, pass a DSN:

```python
Learner(store="postgresql://user:pass@host/db")  # pip install neuraminerl[postgres]
```

`PostgresStore` implements the same 16-method `Store` protocol — JSONB columns, `BYTEA`
embeddings, numpy cosine search, and no in-process cache, so concurrent workers see each other's
lessons immediately. It pools connections, so a Postgres restart or network blip reconnects
instead of breaking the process. pgvector is an optimization for far larger stores, not a
requirement. Implement `Store` yourself for MySQL, Mongo, or a hosted vector DB.

Embedding width is enforced per scope rather than per database, so agents sharing one Postgres
may use different embedders — a worker without `model2vec` falling back to the hashed embedder
cannot lock the others out.

**Async agents.** `AsyncLearner` mirrors the sync API but offloads every blocking call (store IO
and the reflection call inside `end()`) to a worker thread, so LangGraph nodes and FastAPI
handlers never block the event loop:

```python
from neuraminerl import AsyncLearner

nm = AsyncLearner(scope="checkout-agent")
async with await nm.run(task="...") as run:
    prompt = SYSTEM_PROMPT + str(await run.lessons())  # awaitable here, a property on Run
    await run.end(success=True)
```

**Your own cost tracking.** `on_usage` fires after every reflection and dedup call:

```python
Learner(on_usage=lambda e: record_llm_usage(e.model, e.input_tokens, e.output_tokens))
```

A callback that raises is reported as a warning and never breaks the run.

## What does it cost in tokens?

Injection is bounded, not cumulative — a sliding window, never a snowball:

| Cost | When | Size |
| --- | --- | --- |
| Lesson block (input tokens) | runs with relevant lessons | typically 150–300 tokens, hard-capped at `token_budget` (default 800) |
| Reflection call | failures only, off the hot path | ~2k input (transcript capped) + a few lessons out |
| Dedup merge call | only when a new lesson overlaps an existing one | tiny |
| Retrieval embeddings | every run | zero — embeddings run locally |

The comparison that matters is not "800 tokens vs 0" — it's "800 tokens vs the cost of
repeating failures." A failed agentic run wastes its entire token spend plus a retry. If a run
costs ~10k tokens, injection overhead is ~2–3%, and preventing one failure per ~40 runs breaks
even; everything past that is profit. Check what you're actually paying with
`run.lessons.token_count` (this run) and `nm.stats().injected_tokens_estimate` (cumulative).

Two tuning notes:

- **Prompt caching**: place the lesson block *after* your static system prompt (or in the first
  user message), not before it — a varying block early in the prompt invalidates the provider's
  prompt-cache for everything behind it, which costs far more than the block itself. Between
  failures the block is byte-identical, so positioned correctly it caches too.
- **Budget by task type**: short decision tasks (classification, routing) rarely need more than
  2 lessons — `Learner(k=2, token_budget=300)`. The defaults suit longer agentic runs.

## Core API

| Call | Purpose |
| --- | --- |
| `Learner()` | Zero-config init. `Learner(scope="checkout-agent", llm="anthropic:claude-haiku-4-5", store="postgresql://...", on_usage=...)` to customize. |
| `AsyncLearner()` | Same API, thread-offloaded for asyncio callers. `await run.lessons()` replaces the property. |
| `nm.run(task=...)` | Context manager. Yields a `Run`; unhandled exceptions become failures. |
| `run.lessons` | Recalled lessons for this task; `str()` renders the injectable prompt block. Recall through the run binds lessons for credit assignment. |
| `run.log(messages)` / `run.log_tool_call(...)` | Best-effort trajectory capture. |
| `run.end(success=..., error=..., score=...)` | Record the outcome; triggers reflection on failure. |
| `nm.feedback(run_id, note, success=...)` | Delayed outcome ("user said this was wrong two hours later"). |
| `nm.lessons()` / `nm.forget(lesson_id)` | Audit and control what gets injected. |
| `nm.stats()` | Baseline success rate, lesson counts by state, cumulative injected-token estimate. |

The three IO boundaries are small `Protocol`s you can implement yourself and pass to `Learner`:
`Store` (persistence + vector search), `Embedder` (retrieval vectors), and `LLMClient`
(reflection). Reflection, retrieval, injection, and the lesson lifecycle are internal classes
tuned entirely through `LearnerConfig`.

## Status

Early alpha — the API may still change. See `examples/` for a runnable demo where an agent
measurably improves across episodes against an API with undocumented quirks.

## License

Apache-2.0
