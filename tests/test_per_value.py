"""Per-value relations: ForEach and parametric trace slicing."""

from __future__ import annotations

from contragent.formulas._pred_key import pred_key
from contragent.formulas.dfa_evaluator import ParametricEvaluator, make_evaluator
from contragent.formulas.evaluator import evaluate, evaluate_final
from contragent.formulas.formula import ForEach
from contragent.formulas.parser import parse_repr
from contragent.formulas.sat import is_satisfiable
from contragent.generation.structured_ir import ConstraintIR, compile_ir

COMMENT, LIKE = "venmo__create_transaction_comment", "venmo__like_transaction"
REMIND = "venmo__remind_payment_request"


def call(tool, **args):
    v = {pred_key("called", tool): True, pred_key("called_any"): True}
    for k, x in args.items():
        v[pred_key("arg_value", tool, k)] = x
    return v


def per(relation, subject, obj=None, quantifier=None, field="id"):
    res = compile_ir(
        ConstraintIR(
            subject=subject,
            object=obj,
            relation=relation,
            scope="global",
            quantifier=quantifier,
            params={"per": field},
        )
    )
    assert not res.error, res.error
    return res.compiled.formula


def test_comment_before_like_on_the_same_transaction():
    f = per("precedes", COMMENT, LIKE)
    assert isinstance(f, ForEach)
    assert evaluate(f, [call(COMMENT, id=1), call(LIKE, id=1)])
    # A comment on another transaction does not license this like.
    assert not evaluate(f, [call(COMMENT, id=2), call(LIKE, id=1)])
    # Ids compare as strings: 1 and "1" are one transaction.
    assert evaluate(f, [call(COMMENT, id="1"), call(LIKE, id=1)])


def test_the_monitor_slices_by_value():
    m = make_evaluator(per("precedes", COMMENT, LIKE))
    assert isinstance(m, ParametricEvaluator)
    assert m.step(call(COMMENT, id=1)) == "?"
    assert m.step(call("other_tool")) == "?"  # belongs to no slice
    assert m.step(call(LIKE, id=1)) == "?"
    snap = m.snapshot()
    assert m.step(call(LIKE, id=2)) == "⊥"
    m.restore(snap)  # a probe that is rolled back
    assert m.peek() == "?"


def test_at_most_once_per_request_not_once_overall():
    m = make_evaluator(per("limits", REMIND, quantifier=1, field="payment_request_id"))
    assert m.step(call(REMIND, payment_request_id=1608)) == "?"
    assert m.step(call(REMIND, payment_request_id=1612)) == "?"
    assert m.step(call(REMIND, payment_request_id=1608)) == "⊥"


def test_every_app_logged_into_is_logged_out_of():
    f = per("follows", "login", "logout", field="app")
    trace = [call("login", app="spotify"), call("login", app="venmo"), call("logout", app="venmo")]
    assert not evaluate_final(f, trace)  # spotify still open
    assert evaluate_final(f, trace + [call("logout", app="spotify")])
    m = make_evaluator(f)
    for v in trace:
        m.step(v)
    assert m.finalize() == "⊥"


def test_repr_round_trips_and_sat_reads_the_body():
    f = per("follows", "login", "logout", field="app")
    assert parse_repr(repr(f)) == f
    assert is_satisfiable([f]) is not False
    ban = compile_ir(ConstraintIR(subject="logout", relation="bans", scope="global")).compiled
    assert is_satisfiable([per("precedes", COMMENT, LIKE), ban.formula]) is not False


def test_per_field_by_tool_and_unsupported_relations():
    res = compile_ir(
        ConstraintIR(
            subject=COMMENT,
            object=LIKE,
            relation="precedes",
            scope="global",
            params={"per": {COMMENT: "transaction_id", LIKE: "id"}},
        )
    )
    f = res.compiled.formula
    assert evaluate(f, [call(COMMENT, transaction_id=5), call(LIKE, id=5)])
    bad = compile_ir(
        ConstraintIR(subject=LIKE, relation="bans", scope="global", params={"per": "id"})
    )
    assert bad.compiled is None and "not supported" in bad.error
    missing = compile_ir(
        ConstraintIR(
            subject=COMMENT,
            object=LIKE,
            relation="precedes",
            scope="global",
            params={"per": {COMMENT: "id"}},
        )
    )
    assert "names no field" in missing.error
