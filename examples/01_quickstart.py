"""The README snippet, runnable. Run it twice:

    python examples/01_quickstart.py   # the "agent" fails; neuraminerl reflects
    python examples/01_quickstart.py   # the lesson is injected; it succeeds

Works best with ANTHROPIC_API_KEY or OPENAI_API_KEY set (real reflection).
Without a key it still works, storing the raw failure as an observation.
Delete ./.neuraminerl to reset.
"""

from __future__ import annotations

from neuraminerl import Learner

SYSTEM_PROMPT = "You are a booking agent.\n"


def my_agent(prompt: str) -> tuple[bool, str | None]:
    """A stand-in agent with one hidden failure mode: it fails unless the
    prompt warns it about the date format."""
    if "YYYY-MM-DD" in prompt or "ISO" in prompt or "failed" in prompt:
        return True, None
    return False, "booking form rejected: invalid value for field 'date'"


def main() -> None:
    nm = Learner()

    with nm.run(task="Book the cheapest NYC->SFO flight on the demo site") as run:
        prompt = SYSTEM_PROMPT + str(run.lessons)  # inject lessons from past failures
        ok, error = my_agent(prompt)
        run.log({"role": "assistant", "content": f"submitting booking form... ok={ok}"})
        run.end(success=ok, error=error)

    if run.lessons:
        print("lessons injected this run:")
        print(str(run.lessons))
    print("outcome:", "success" if ok else f"failure ({error})")

    if not ok:
        print("\nneuraminerl reflected on the failure. stored lessons:")
        for lesson in nm.lessons():
            print(f"  - {lesson.text}")
        print("\nrun me again — the agent will get these lessons up front.")


if __name__ == "__main__":
    main()
