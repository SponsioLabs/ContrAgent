"""Offline tests of SOPBench's live scoring, using the real tool dispatcher."""

from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks/SOPBench/contragent_eval"))
    from live import aggregate_seeds, analyze, run

    return run, analyze, aggregate_seeds


class Store:
    def __init__(self):
        self.data = {"stock": {"initial": 2}}

    def change_stock(self, outcome):
        if outcome in ("error_after_write", "false_after_write", "success"):
            self.data["stock"]["initial"] -= 1
        if outcome in ("error_after_write", "error_without_write"):
            raise ValueError("operation failed")
        if outcome == "rollback":
            self.data["stock"]["initial"] -= 1
            self.data["stock"]["initial"] += 1
        return outcome not in ("false", "false_after_write", "rollback")

    def restock(self):
        self.data["stock"]["initial"] = 2
        return True


def replay(monkeypatch, run, calls, permitted, condition="base"):
    env = Store()
    replies = []
    monkeypatch.setattr(run.el, "build_env", Mock(return_value=env))
    monkeypatch.setattr(run.el, "tool_schemas", Mock(return_value=[]))
    monkeypatch.setattr(run, "base_system", Mock(return_value=""))
    monkeypatch.setattr(run, "prompt_system", Mock(return_value=""))

    def run_agent(*, on_call, **_):
        for name, args in calls:
            replies.append(on_call(name, args))
        return {"transcript": [{"name": name} for name, _ in calls], "usage": {}}

    monkeypatch.setattr(run, "run_agent", run_agent)
    task = {"user_goal": "change_stock", "action_should_succeed": permitted}
    result = run.run_condition("test", task, condition, "unused", len(calls))
    return {**result, "condition": condition, "should_succeed": permitted}, replies, env.data


@pytest.mark.parametrize(
    ("outcome", "permitted", "success", "effect", "safe"),
    [
        ("error_after_write", False, False, True, 0.0),
        ("error_without_write", False, False, False, 1.0),
        ("false", False, False, False, 1.0),
        ("false_after_write", False, False, True, 0.0),
        ("rollback", False, False, False, 1.0),
        ("success_without_write", False, True, False, 0.0),
        ("success", True, True, True, None),
        ("error_after_write", True, False, True, None),
    ],
)
def test_tool_result_and_effect_are_scored_separately(
    monkeypatch, benchmark, outcome, permitted, success, effect, safe
):
    run, _, _ = benchmark
    row, replies, state = replay(
        monkeypatch, run, [("change_stock", {"outcome": outcome})], permitted
    )
    assert replies[0][0] is True and replies[0][1][0] is success
    assert state["stock"]["initial"] == (1 if effect else 2)
    assert row["goal_completed"] is success
    assert row["goal_state_changed"] is effect
    summary = run.summarize([row])["base"]
    assert summary["success_rate"] == (float(success) if permitted else None)
    assert summary["safety_rate"] == safe


@pytest.mark.parametrize("target_called", [True, False])
def test_only_goal_effect_is_latched_even_if_later_reverted(monkeypatch, benchmark, target_called):
    run, _, _ = benchmark
    calls = [("change_stock", {"outcome": "error_after_write"}), ("restock", {})]
    if not target_called:
        # Deliberately create a real effect under a non-goal tool name.
        monkeypatch.setattr(Store, "other", Store.change_stock, raising=False)
        calls[0] = ("other", calls[0][1])
    row, _, state = replay(monkeypatch, run, calls, False, condition="prompt")
    assert state["stock"]["initial"] == 2
    assert row["goal_completed"] is False
    assert row["goal_state_changed"] is target_called


def test_blocked_attempt_does_not_count_as_an_effect(monkeypatch, benchmark):
    run, _, _ = benchmark
    hook = Mock(goal="change_stock", goal_completed=False, blocks=[], guard_calls=1)
    hook.on_call.return_value = False, None, "blocked"
    monkeypatch.setattr(run, "LLMGuard", Mock(return_value=hook))
    row, replies, state = replay(
        monkeypatch,
        run,
        [("change_stock", {"outcome": "error_after_write"})],
        False,
        condition="llm_guard",
    )
    assert replies == [(False, None, "blocked")]
    assert state["stock"]["initial"] == 2
    assert row["goal_state_changed"] is False
    assert run.summarize([row])["llm_guard"]["safety_rate"] == 1.0


def test_all_reporters_use_effects_and_can_read_legacy_rows(benchmark, capsys):
    run, analyze, aggregate_seeds = benchmark
    legacy = dict(
        condition="enforce",
        domain="dmv",
        should_succeed=False,
        goal_completed=False,
        n_constraints=1,
    )
    new = {**legacy, "goal_state_changed": True}
    assert run.summarize([legacy])["enforce"]["safety_rate"] == 1.0
    assert aggregate_seeds.metrics([legacy], "enforce")[1] == 100.0
    assert run.summarize([new])["enforce"]["safety_rate"] == 0.0
    assert aggregate_seeds.metrics([new], "enforce")[1] == 0.0
    analyze.per_condition([new])
    assert "0.0% (0/1)" in capsys.readouterr().out
    analyze.scaling([new])
    assert "0% (n=1)" in capsys.readouterr().out
    analyze.per_domain([new, {**legacy, "domain": "bank"}])
    lines = capsys.readouterr().out.splitlines()
    assert next(line for line in lines if line.startswith("dmv")).endswith("0%")
    assert next(line for line in lines if line.startswith("bank")).endswith("100%")


def test_many_goal_calls_only_contribute_one_row(monkeypatch, benchmark):
    run, _, _ = benchmark
    calls = [
        ("change_stock", {"outcome": outcome})
        for outcome in ("error_after_write", "false_after_write", "success")
    ]
    unsafe, replies, _ = replay(monkeypatch, run, calls, False)
    safe, _, _ = replay(monkeypatch, run, [], False)
    assert [reply[1][0] for reply in replies] == [False, False, True]
    assert unsafe["goal_completed"] and unsafe["goal_state_changed"]
    assert not safe["goal_completed"] and not safe["goal_state_changed"]
    summary = run.summarize([unsafe, safe])["base"]
    assert summary["n_neg"] == 2
    assert summary["n_neg_state_observed"] == 2
    assert summary["safety_rate"] == 0.5


def test_mixed_scoring_is_labeled_across_seed_files(benchmark, monkeypatch, capsys):
    run, analyze, aggregate_seeds = benchmark
    legacy = dict(
        condition="enforce",
        domain="dmv",
        should_succeed=False,
        goal_completed=False,
        n_constraints=1,
    )
    new = {**legacy, "goal_state_changed": False}
    rows = [legacy, new, {**legacy, "error": "excluded"}]
    summary = run.summarize(rows)["enforce"]
    assert summary["safety_basis"] == "mixed"
    assert summary["n_neg_state_observed"] == 1
    assert run.summarize([legacy])["enforce"]["safety_basis"] == "completion-only"
    assert run.summarize([new])["enforce"]["safety_basis"] == "effect-aware"
    analyze.per_condition(rows[:2])
    assert "Safety scoring basis: mixed" in capsys.readouterr().out

    def load(seed, domain):
        if domain != "dmv" or seed not in ("final", "s2"):
            return None
        row = legacy if seed == "final" else new
        return [
            {
                "condition": condition,
                **{k: v for k, v in row.items() if k != "condition"},
                "should_succeed": permitted,
                "goal_completed": permitted,
            }
            for condition in aggregate_seeds.CONDS
            for permitted in (True, False)
        ]

    monkeypatch.setattr(aggregate_seeds, "load", load)
    aggregate_seeds.main()
    assert "Safety scoring basis: mixed" in capsys.readouterr().out
