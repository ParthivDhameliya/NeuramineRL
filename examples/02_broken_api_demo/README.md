# The broken-API demo

An agent places orders against a mock e-commerce API with **five undocumented
quirks** (ISO dates, integer-cent amounts, a required `Idempotency-Key`
header, uppercase ISO-2 country codes, and 409s that must be retried with a
*fresh* key). The API docs the agent sees deliberately omit all of this.

Two arms run the same 10 episodes x 5 orders:

- **baseline** — the agent starts amnesiac every episode and keeps failing
  the same ways forever.
- **neuramine** — failures are reflected into conditioned lessons, lessons
  are injected into later runs, and lesson health is tracked against the
  agent's baseline.

```bash
python demo.py            # offline: deterministic, no API keys (what CI runs)
python demo.py --live     # a real LLM plays the agent and writes reflections
python demo.py --check    # exit 1 unless the learning curve improves >= 40 pts
```

Offline mode uses a scripted agent and scripted reflection so the run is
deterministic — it proves the *pipeline* (capture -> reflect -> dedup ->
inject -> credit -> lifecycle), not the LLM. `--live` is the honest
experiment.

Expected offline output: baseline flat at 0%, neuramine climbing to 100% by
episode 3, and exactly five active lessons at the end.
