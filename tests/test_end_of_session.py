"""The verdict of a contract when the session ends.

An obligation still pending at the end (the next event of an ``X``, the
event an ``F`` or a ``U`` waits for) is read on the idle events after
the last one: no tool is called and the state stays what it was. A ``G``
binds the events of the session. The runtime monitor
(``DFAEvaluator.finalize``), the offline replay (``evaluate_final``), and
the satisfiability search (``sat``) must agree with each other and with
the reference below on every formula and trace.
"""

from __future__ import annotations

import copy
import itertools
import random

import pytest

from contragent.formulas import sat
from contragent.formulas._pred_key import idle_valuation
from contragent.formulas.dfa_evaluator import DFAEvaluator
from contragent.formulas.evaluator import evaluate_final
from contragent.formulas.formula import And, Atom, F, G, Implies, Not, Or, U, X
from contragent.formulas.parser import parse_repr
from contragent.formulas.sat import is_satisfiable
from contragent.models.trace import Event, Trace
from contragent.tracer.grounding import GroundingState, ground_event

CALL_A, CALL_B = Atom("called", "a"), Atom("called", "b")
AUTH = Atom("ctx", "auth", "yes")  # a fact: part of the state
ATOMS = (CALL_A, CALL_B, AUTH)
KEYS = tuple(a.key() for a in ATOMS)


def reference(formula, trace):
    """The verdict read directly: position ``n`` stands for every idle event."""
    n = len(trace)
    idle = idle_valuation(trace[-1]) if trace else {}

    def at(i):
        return trace[i] if i < n else idle

    def holds(f, i):
        if isinstance(f, Atom):
            return bool(at(i).get(f.key(), False))
        if isinstance(f, Not):
            return not holds(f.child, i)
        if isinstance(f, And):
            return holds(f.left, i) and holds(f.right, i)
        if isinstance(f, Or):
            return holds(f.left, i) or holds(f.right, i)
        if isinstance(f, Implies):
            return not holds(f.left, i) or holds(f.right, i)
        if isinstance(f, X):
            return holds(f.child, min(i + 1, n))
        if isinstance(f, G):
            return all(holds(f.child, j) for j in range(i, n))
        if isinstance(f, F):
            return any(holds(f.child, j) for j in range(i, n + 1))
        if isinstance(f, U):
            return any(
                holds(f.right, j) and all(holds(f.left, k) for k in range(i, j))
                for j in range(i, n + 1)
            )
        raise TypeError(f)

    return holds(formula, 0)


def runtime(formula, trace):
    dfa = DFAEvaluator(formula)
    for v in trace:
        dfa.step(v)
    return dfa.finalize() == "⊤"


def search(formula, trace):
    """The satisfiability search's reading: its own abstraction and progression."""
    table = sat._LeafTable()
    node = sat._abstract(formula, table)

    def val(v):
        return frozenset(table.atom_ids[k] for k, b in v.items() if b and k in table.atom_ids)

    for v in trace:
        node = sat._progress(node, val(v))
    return sat._finalize(node, sat._idle_props(table, val(trace[-1] if trace else {}))) is True


def _random_formula(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(ATOMS)
    k = rng.choice("NAOIXGFU")
    if k == "N":
        return Not(_random_formula(rng, depth - 1))
    if k in "AOIU":
        cls = {"A": And, "O": Or, "I": Implies, "U": U}[k]
        return cls(_random_formula(rng, depth - 1), _random_formula(rng, depth - 1))
    return {"X": X, "G": G, "F": F}[k](_random_formula(rng, depth - 1))


LETTERS = [
    {k: True for k in combo}
    for r in range(len(KEYS) + 1)
    for combo in itertools.combinations(KEYS, r)
]


def test_every_reading_agrees_with_the_reference():
    rng = random.Random(7)
    for _ in range(1500):
        f = _random_formula(rng, 3)
        for n in range(4):
            trace = [rng.choice(LETTERS) for _ in range(n)]
            want = reference(f, trace)
            assert evaluate_final(f, trace) == want, (f, trace)
            assert runtime(f, trace) == want, (f, trace)
            assert search(f, trace) == want, (f, trace)


# The cases the semantics was settled on, a session ending right after the
# trigger unless the trace says otherwise.
CASES = [
    # an obligation to act fails, a prohibition holds
    ("G((called('modify') -> X(called('get'))))", [["modify"]], False),
    ("G((called('modify') -> X(called('get'))))", [["modify"], ["get"]], True),
    ("G((called('send') -> X((!(called('send')) & X(!(called('send')))))))", [["send"]], True),
    (
        "G((called('send') -> X((!(called('send')) & X(!(called('send')))))))",
        [["send"], ["read"]],
        True,
    ),
    # stopping early earns nothing: within two events, or two events on
    ("G((called('modify') -> (X(called('get')) | X(X(called('get'))))))", [["modify"]], False),
    (
        "G((called('modify') -> (X(called('get')) | X(X(called('get'))))))",
        [["modify"], ["other"], ["get"]],
        True,
    ),
    ("G((called('modify') -> X(X(called('get')))))", [["modify"]], False),
    ("G((called('modify') -> X(X(called('get')))))", [["modify"], ["other"]], False),
    # a fact keeps the value it had at the last event
    ("G((called('login') -> X(ctx('auth', 'yes'))))", [["login", "ctx:auth"]], True),
    ("G((called('login') -> X(ctx('auth', 'yes'))))", [["login"]], False),
    # a G binds the events of the session only
    ("G(called('a'))", [["a"], ["a"]], True),
]


def _valuation(names):
    v = {}
    for name in names:
        if name.startswith("ctx:"):
            v[f"ctx({name[4:]}, yes)"] = True
        else:
            v[f"called({name})"] = True
    return v


@pytest.mark.parametrize(("src", "events", "want"), CASES)
def test_settled_cases(src, events, want):
    f = parse_repr(src)
    trace = [_valuation(e) for e in events]
    assert reference(f, trace) is want
    assert evaluate_final(f, trace) is want
    assert runtime(f, trace) is want
    assert search(f, trace) is want


def test_satisfiability_needs_the_obligation_met_within_the_trace():
    after = parse_repr("G((called('modify') -> X(called('get'))))")
    assert is_satisfiable([after, F(Atom("called", "modify"))]) is True
    assert (
        is_satisfiable([after, F(Atom("called", "modify")), G(Not(Atom("called", "get")))]) is False
    )
    no_repeat = parse_repr("G((called('send') -> X(!(called('send')))))")
    assert is_satisfiable([no_repeat, F(Atom("called", "send"))]) is True


def test_the_idle_event_is_what_grounding_emits_for_an_event_of_no_kind():
    trace = Trace(
        events=[
            Event(ts=1, event_type="context_update", agent="bot", args={"auth": "yes"}),
            Event(ts=2, event_type="tool_call", agent="bot", tool="pay", args={"amount": 5}),
            Event(ts=3, event_type="llm_response", agent="bot", content="done"),
            Event(ts=4, event_type="tool_call", agent="bot", tool="refund", args={"id": "x"}),
        ]
    )
    content = {"arg_numeric": {("pay", "amount")}, "llm_said": {("done",)}}
    state = GroundingState()
    for i, event in enumerate(trace.events):
        last = ground_event(event, i, state, content_atoms=content)
        nothing = Event(ts=event.ts, event_type="context_update", agent="bot")
        assert ground_event(
            nothing, i, copy.deepcopy(state), content_atoms=content
        ) == idle_valuation(last)
