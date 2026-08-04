"""The broken-API demo: watch an agent learn an API's undocumented quirks.

Two arms run the same 10 episodes x 5 order tasks:

- baseline:  the agent starts amnesiac every episode.
- neuramine: failures are reflected into lessons, lessons are injected into
  later episodes, and lesson health is tracked.

Offline (default, no keys needed): a deterministic scripted agent + scripted
reflection exercise the full neuramine pipeline. Live (--live, needs
ANTHROPIC_API_KEY or OPENAI_API_KEY): a real LLM plays the agent and writes
the reflections.

Usage:
    python demo.py [--live] [--episodes 10] [--check]
"""

from __future__ import annotations

import argparse
import sys

from agents import EpisodeResult, LiveAgent, ScriptedAgent, ScriptedReflectionLLM
from mock_api import TASKS, OrdersAPI, OrderTask

from neuramine import Learner, Run


def run_arm(
    label: str, episodes: int, *, learner: Learner | None, agent: ScriptedAgent | LiveAgent
) -> list[float]:
    """Returns per-episode success rate. learner=None means the baseline arm."""
    rates: list[float] = []
    for episode in range(episodes):
        api = OrdersAPI()  # fresh API state (and fresh 409s) each episode
        successes = 0
        for task in TASKS:
            if learner is None:
                result = _attempt_without_memory(agent, task, api)
            else:
                with learner.run(task=task.describe()) as run:
                    result = agent.attempt(task, api, str(run.lessons), run)
                    run.end(success=result.ok, error=result.error)
            successes += int(result.ok)
        rate = successes / len(TASKS)
        rates.append(rate)
        bar = "#" * round(rate * 20)
        print(f"  {label} episode {episode + 1:>2}: {rate:>5.0%} {bar}")
    return rates


def _attempt_without_memory(
    agent: ScriptedAgent | LiveAgent, task: OrderTask, api: OrdersAPI
) -> EpisodeResult:
    """Baseline arm: same agent, no lessons, throwaway capture."""
    throwaway = Learner(
        store=":memory:", embedder="hashed", llm=ScriptedReflectionLLM(), reflect="off"
    )
    run: Run = throwaway.run(task=task.describe())
    with run:
        result = agent.attempt(task, api, "", run)
        run.end(success=result.ok, error=result.error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use a real LLM from env keys")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 unless the learning-curve assertion holds"
    )
    args = parser.parse_args()

    if args.live:
        from neuramine.llm import auto

        llm = auto.detect()
        if llm is None:
            print("--live needs ANTHROPIC_API_KEY or OPENAI_API_KEY")
            return 2
        agent: ScriptedAgent | LiveAgent = LiveAgent(llm)
        learner = Learner(store=":memory:", llm=llm)
    else:
        agent = ScriptedAgent()
        learner = Learner(store=":memory:", embedder="hashed", llm=ScriptedReflectionLLM())

    print(f"\n=== baseline (no memory{', live' if args.live else ''}) ===")
    baseline = run_arm("baseline ", args.episodes, learner=None, agent=agent)

    print(f"\n=== neuramine (learning{', live' if args.live else ''}) ===")
    learning = run_arm("neuramine", args.episodes, learner=learner, agent=agent)

    print("\n=== lessons learned ===")
    for lesson in learner.lessons():
        mark = {"active": "+", "candidate": "?", "retired": "-", "merged": "~"}[lesson.state]
        print(f"  [{mark} {lesson.state:<9}] {lesson.text}")

    stats = learner.stats()
    print(f"\nbaseline success rate (rolling): {stats.baseline_success_rate:.0%}")

    early = sum(learning[:2]) / 2
    late = sum(learning[-4:]) / 4
    improvement = late - early
    flat = (sum(baseline[-4:]) / 4) - (sum(baseline[:2]) / 2)
    print(f"neuramine arm improvement (late - early): {improvement:+.0%}")
    print(f"baseline arm improvement (late - early):  {flat:+.0%}")

    if args.check and improvement < 0.4:
        print("FAIL: expected >= 40 point improvement with learning enabled")
        return 1
    print("\nOK" if improvement >= 0.4 else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
