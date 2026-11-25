"""The live SOPBench hooks read a task's goal action the same way."""

from pathlib import Path

import pytest

TASKS = [
    {"user_goal": "transfer_title"},
    {"directed_action_graph": {"nodes": [["renew_vehicle"], ["login_user"]]}},
]


@pytest.fixture
def live(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks/SOPBench/contragent_eval"))
    from live import enforce, llm_guard, run

    monkeypatch.setattr(llm_guard.sopmod, "render_sop", lambda *_: "")
    monkeypatch.setattr(llm_guard, "_tool_catalog", lambda _: "")
    return enforce, llm_guard, run


@pytest.mark.parametrize("task", TASKS)
def test_hooks_share_the_goal(live, task):
    enforce, llm_guard, run = live
    goal = enforce.goal_action(task)
    assert goal in ("transfer_title", "renew_vehicle")
    assert run.Passthrough("dmv", task, None).goal == goal
    assert llm_guard.LLMGuard("dmv", task, None, api_key="unused").goal == goal
