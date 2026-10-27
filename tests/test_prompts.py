"""Prompts: what the contracts ask of the agent next.

After a call, a guarantee that owes a call at the next event (an ``X``
whose trigger fired) prompts for it; before the session ends,
:meth:`ContrAgent.guard_finish` prompts for what is still owed. Both read
the contracts as the end of the session does: after the last event
nothing happens.
"""

from __future__ import annotations

from contragent import ContrAgent

VERIFY = {
    "guarantee": "G((called('modify') -> X(called('get_order_details'))))",
    "desc": "modify must be followed by get_order_details verification",
}
NO_REPEAT = {
    "guarantee": "G((called('send') -> X(!(called('send')))))",
    "desc": "no two transfers in a row",
}
CLOSE = {
    "guarantee": "G((called('open_case') -> F(called('close_case'))))",
    "desc": "an opened case is closed",
}
TOOLS = ("modify", "get_order_details", "send", "open_case", "close_case", "other")


def _session(*contracts, mode="gate"):
    guard = ContrAgent("bot", contracts=list(contracts), mode=mode)
    tools = guard.wrap({t: (lambda t=t: f"{t} done") for t in TOOLS})
    return guard, tools


def test_a_call_that_owes_the_next_one_prompts_for_it():
    guard, tools = _session(VERIFY, NO_REPEAT)
    assert tools["modify"]() == (
        "modify done\n\nThe action `modify` was completed. Policy: modify must be followed by "
        "get_order_details verification. Call `get_order_details` next."
    )
    assert guard.guard_before("other").blocked
    assert tools["get_order_details"]() == "get_order_details done"


def test_a_prohibition_prompts_nothing():
    guard, tools = _session(NO_REPEAT)
    assert tools["send"]() == "send done"
    assert guard.guard_finish().allowed


def test_the_end_prompts_for_what_is_still_owed():
    guard, tools = _session(CLOSE)
    assert tools["open_case"]() == "open_case done"
    check = guard.guard_finish()
    assert not check.allowed
    assert check.prompts == [
        "The session cannot end yet. Policy: an opened case is closed. "
        "Call `close_case` before finishing."
    ]
    tools["close_case"]()
    assert guard.guard_finish().allowed
    assert guard.finish_session() == []


def test_the_end_prompts_a_bounded_number_of_times():
    guard, tools = _session(CLOSE)
    tools["open_case"]()
    assert not guard.guard_finish(max_prompts=2).allowed
    assert not guard.guard_finish(max_prompts=2).allowed
    assert guard.guard_finish(max_prompts=2).allowed
    # The agent did not comply: the end decides it as a violation.
    assert [v.desc for v in guard.finish_session()] == ["an opened case is closed"]


def test_an_owed_next_call_is_decided_at_the_end():
    guard, tools = _session(VERIFY)
    tools["modify"]()
    assert [v.desc for v in guard.finish_session()] == [VERIFY["desc"]]


def test_flag_mode_does_not_prompt():
    guard, tools = _session(VERIFY, CLOSE, mode="flag")
    assert tools["modify"]() == "modify done"
    tools["open_case"]()
    assert guard.guard_finish().allowed


def test_probing_leaves_the_grounding_misses_alone():
    from contragent.tracer.grounding import grounding_misses

    guard, tools = _session(
        VERIFY,
        {"guarantee": "G((called('send') -> !(arg_field_has('send', 'to', 'evil'))))", "desc": "x"},
    )
    before = grounding_misses()
    assert "Call `get_order_details` next." in tools["modify"]()
    assert guard.guard_finish().prompts
    assert grounding_misses() == before
