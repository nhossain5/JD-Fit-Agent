"""
Evals for the JD Fit Agent.

A tiny labeled test set + a runner that checks whether the agent correctly
flags the skills we KNOW should be missing for each (resume, JD) pair.

Why this exists: without it, you're eyeballing every run. This catches
regressions automatically -- e.g., the false-positive "no gaps" we hit when
skill extraction was too narrow. If someone breaks extraction, a case fails.

This is a lightweight STRING-based eval: does the agent's answer mention the
expected missing skills? A next-level version would check structured output or
use an LLM as a judge. Also note: the agent uses an LLM, so results are
probabilistic -- track the pass RATE over time, don't expect identical runs.
"""

import os
import agent  # reuse the agent's functions; importing does NOT run its __main__

# Isolate evals from your real memory file so we never touch agent_memory.json.
agent.MEMORY_FILE = "eval_memory.json"

TASK = (
    "A resume and a job description are stored in memory. "
    "Which required skills am I missing, and how is my overall fit?"
)

# Each case: a resume, a JD, and the skills we KNOW should be flagged missing.
CASES = [
    {
        "name": "missing Docker",
        "resume": "Python developer experienced with React and REST APIs.",
        "jd": "We need a Python engineer who knows React and Docker for deployment.",
        "expect_missing": ["Docker"],
    },
    {
        "name": "missing AWS and Kubernetes",
        "resume": "Backend engineer skilled in Python and PostgreSQL.",
        "jd": "Seeking an engineer with AWS and Kubernetes experience.",
        "expect_missing": ["AWS", "Kubernetes"],
    },
    {
        "name": "no gaps",
        "resume": "Full-stack engineer with Java, SQL, and Git.",
        "jd": "Requires Java, SQL, and Git.",
        "expect_missing": [],
    },
]


def run_case(case: dict) -> str:
    """Load the case's resume + JD into (isolated) memory and run the agent."""
    agent.save_memory({"resume": case["resume"], "current_jd": case["jd"]})
    return agent.run_agent(TASK)


def check(case: dict, answer: str):
    """Pass if every expected-missing skill is mentioned in the answer."""
    ans = (answer or "").lower()
    for skill in case["expect_missing"]:
        if skill.lower() not in ans:
            return False, f"did NOT flag missing skill: {skill}"
    return True, "ok"


def main() -> None:
    passed = 0
    for case in CASES:
        answer = run_case(case)
        ok, reason = check(case, answer)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}: {reason}")
        passed += int(ok)

    print(f"\nScore: {passed}/{len(CASES)} passed")

    # Clean up the throwaway eval memory file.
    if os.path.exists(agent.MEMORY_FILE):
        os.remove(agent.MEMORY_FILE)


if __name__ == "__main__":
    main()