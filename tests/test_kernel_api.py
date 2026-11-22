"""``contragent.kernel``: the names a runtime that owns the agent loop imports,
and one pass through them (ground an event, step a monitor, ask whether the
rest can still be satisfied)."""

import contragent.kernel as K
from contragent import Event
from contragent.formulas import Atom, F, G, Implies

NAMES = {
    "GroundingState",
    "ground_event",
    "collect_content_atoms",
    "pred_key",
    "idle_valuation",
    "DFAEvaluator",
    "ParametricEvaluator",
    "make_evaluator",
    "Progressor",
    "is_satisfiable",
    "completable",
    "agent_controlled_atoms",
    "demands_eventuality",
    "resolve_entry",
    "ParseError",
}


def test_the_kernel_exports_exactly_its_stable_names():
    assert set(K.__all__) == NAMES
    assert all(hasattr(K, n) for n in NAMES)


def test_the_names_are_the_objects_of_the_modules_they_come_from():
    from contragent.analysis.completion import completable
    from contragent.formulas.dfa_evaluator import DFAEvaluator, make_evaluator
    from contragent.formulas.sat import is_satisfiable
    from contragent.tracer.grounding import GroundingState, ground_event

    assert K.completable is completable and K.is_satisfiable is is_satisfiable
    assert K.DFAEvaluator is DFAEvaluator and K.make_evaluator is make_evaluator
    assert K.GroundingState is GroundingState and K.ground_event is ground_event


def test_a_runtime_grounds_steps_and_asks_what_is_left():
    refund_needs_check = G(Implies(Atom("called", "refund"), Atom("fact", "checked")))
    close_owed = F(Atom("called", "close"))
    monitors = [K.make_evaluator(refund_needs_check), K.make_evaluator(close_owed)]
    state = K.GroundingState()

    assert K.pred_key("called", "refund") == Atom("called", "refund").key()
    assert K.demands_eventuality(close_owed) and not K.demands_eventuality(refund_needs_check)

    event = Event(ts=0, agent="a", event_type="tool_call", tool="refund", args={})
    valuation = K.ground_event(event, 0, state)
    assert valuation[K.pred_key("called", "refund")]

    # try the step and take it back: the refund without the check refutes its contract
    snapshots = [m.snapshot() for m in monitors]
    assert monitors[0].step(valuation) == "⊥"
    monitors[1].step(valuation)
    assert K.completable([m.residual for m in monitors]) is False
    for m, snap in zip(monitors, snapshots, strict=True):
        m.restore(snap)

    # with the fact the step is fine, and closing later still satisfies the rest
    valuation[K.pred_key("fact", "checked")] = True
    for m in monitors:
        m.step(valuation)
    assert K.completable([m.residual for m in monitors]) is True
    assert K.pred_key("called", "refund") not in K.idle_valuation(valuation)
