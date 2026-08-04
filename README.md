# Neuramine

**Self-improvement for AI agents.** Neuramine gives your agent the ability to learn from its
past mistakes

Every time your agent fails, Neuramine reflects on the failure and distills it into a
*conditioned lesson* ("When submitting the booking form, use ISO dates; MM/DD/YYYY is silently
rejected"). On future runs, the relevant lessons are retrieved and injected into the prompt.
Crucially, Neuramine then **tracks whether each injected lesson actually improved outcomes** —
lessons that help get promoted, lessons that don't decay and get pruned. No pile of stale
superstitions.

```
run agent → capture trajectory → detect outcome → reflect on failures
     ↑                                                    │
     └── inject lessons ← score & prune ← store lessons ←─┘
```

## Quickstart

```bash
pip install neuramine[embeddings]
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY — used for reflection
```

```python
from neuramine import Learner

nm = Learner()  # zero config: SQLite + local embeddings in ./.neuramine/

with nm.run(task="Book the cheapest NYC->SFO flight on the demo site") as run:
    prompt = SYSTEM_PROMPT + str(run.lessons)  # inject lessons from past failures
    result = my_agent(prompt)  # your agent, unchanged
    run.log(result.messages)  # best-effort trajectory capture
    run.end(success=result.ok, error=result.error)
```

Run it twice. The second run is smarter.

On failure, Neuramine reflects (one cheap LLM call, off the hot path) and stores lessons like:

```
<learned_lessons>
Lessons from previous attempts at similar tasks. Apply them unless clearly
inapplicable to the current situation.
1. When submitting the booking form, use ISO dates (YYYY-MM-DD); MM/DD/YYYY is silently rejected.
2. When an API call returns 409, retry once with a new idempotency key instead of changing the payload.
</learned_lessons>
```

## Why not just a memory library?

Storing lessons is the easy part. The hard parts — the parts Neuramine owns — are:

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

## Core API

| Call | Purpose |
| --- | --- |
| `Learner()` | Zero-config init. `Learner(scope="checkout-agent", llm="anthropic:claude-haiku-4-5", ...)` to customize. |
| `nm.run(task=...)` | Context manager. Yields a `Run`; unhandled exceptions become failures. |
| `run.lessons` | Recalled lessons for this task; `str()` renders the injectable prompt block. Recall through the run binds lessons for credit assignment. |
| `run.log(messages)` / `run.log_tool_call(...)` | Best-effort trajectory capture. |
| `run.end(success=..., error=..., score=...)` | Record the outcome; triggers reflection on failure. |
| `nm.feedback(run_id, note, success=...)` | Delayed outcome ("user said this was wrong two hours later"). |
| `nm.lessons()` / `nm.forget(lesson_id)` | Audit and control what gets injected. |
| `nm.stats()` | Baseline success rate, lesson counts by state, top/bottom lessons. |

Every stage is swappable via small Protocols: `Store`, `Embedder`, `LLMClient`,
`OutcomeDetector`, `Reflector`, `Retriever`, `Injector`.

## Status

Early alpha — API may change before 0.2. See `examples/` for a runnable demo where an agent
measurably improves across episodes against an API with undocumented quirks.

## License

Apache-2.0
