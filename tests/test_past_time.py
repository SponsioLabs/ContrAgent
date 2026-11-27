"""Past-time operators and arithmetic between terms.

* Arithmetic: ``Add/Sub/Mul/Div`` terms inside comparisons.
* Past time: ``Y``, ``S``, ``O``, ``H`` formulas and the ``Prev``, ``At``
  (the value when a condition last held) and ``CountSince`` terms.

Each operator is checked against a reference that reads the definition
straight off the trace, then the two evaluators (recursive and DFA) are
checked to agree on random traces, then the parser, the satisfiability
search and the verifier.
"""

from __future__ import annotations

import random

import pytest

from contragent.formulas import (
    Add,
    At,
    CountSince,
    Div,
    H,
    Mul,
    O,
    Prev,
    S,
    Sub,
    Y,
)
from contragent.formulas.dfa_evaluator import DFAEvaluator
from contragent.formulas.evaluator import evaluate, evaluate_final
from contragent.formulas.formula import (
    And,
    Atom,
    Const,
    Eq,
    F,
    G,
    Ge,
    Gt,
    Implies,
    Le,
    Lt,
    Not,
    Or,
    U,
    Var,
    X,
)
from contragent.formulas.nl_gen import formula_to_nl
from contragent.formulas.parser import parse_formula, parse_repr
from contragent.formulas.past import PastTracker, annotate, past_nodes

p = Atom("called", "p")
q = Atom("called", "q")
r = Atom("called", "r")
price = Var("price")
qty = Var("qty")


def _ev(p_=False, q_=False, r_=False, price_=None, qty_=None) -> dict:
    ev: dict = {"called(p)": p_, "called(q)": q_, "called(r)": r_}
    if price_ is not None:
        ev["price"] = price_
    if qty_ is not None:
        ev["qty"] = qty_
    return ev


def _dfa_holds(formula, trace) -> bool:
    dfa = DFAEvaluator(formula)
    v = "?"
    for ev in trace:
        v = dfa.step(ev)
    if v == "?":
        v = dfa.finalize()
    return v == "⊤"


def _rand_trace(rng: random.Random, n: int) -> list[dict]:
    return [
        _ev(
            rng.random() < 0.4,
            rng.random() < 0.3,
            rng.random() < 0.5,
            rng.choice([None, 90, 95, 100, 105, 110]),
            rng.choice([None, 0, 1, 2, 5]),
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Arithmetic between terms
# ---------------------------------------------------------------------------


class TestArith:
    def test_operators(self):
        ev = [{"price": 100, "qty": 4}]
        assert evaluate(Eq(Add(price, qty), Const(104)), ev)
        assert evaluate(Eq(Sub(price, qty), Const(96)), ev)
        assert evaluate(Eq(Mul(price, qty), Const(400)), ev)
        assert evaluate(Eq(Div(price, qty), Const(25)), ev)

    def test_dunders_build_the_same_terms(self):
        assert price + qty == Add(price, qty)
        assert price * 2 == Mul(price, Const(2))
        assert 2 * price == Mul(Const(2), price)
        assert 100 - price == Sub(Const(100), price)

    def test_division_by_zero_has_no_value(self):
        ev = [{"price": 100, "qty": 0}]
        assert evaluate(Gt(Div(price, qty), Const(0)), ev) is False
        assert evaluate(Le(Div(price, qty), Const(0)), ev) is False

    def test_text_numbers_read_as_numbers(self):
        assert evaluate(Eq(Add(Const("1"), Const("2")), Const(3)), [{}])

    def test_bad_op(self):
        from contragent.formulas.formula import Arith

        with pytest.raises(ValueError):
            Arith("%", price, qty)

    def test_product_limit_in_dfa(self):
        # price * qty <= 1000 at every step
        f = G(Le(Mul(price, qty), Const(1000)))
        assert _dfa_holds(f, [{"price": 100, "qty": 5}, {"price": 200, "qty": 5}])
        assert not _dfa_holds(f, [{"price": 100, "qty": 5}, {"price": 300, "qty": 5}])


# ---------------------------------------------------------------------------
# Past-time formulas against their definitions
# ---------------------------------------------------------------------------


def _ref_formula(f, trace, i) -> bool:
    """A past-time formula at event i, read off the definition."""
    if isinstance(f, Atom):
        return bool(trace[i].get(f.key(), False))
    if isinstance(f, Not):
        return not _ref_formula(f.child, trace, i)
    if isinstance(f, And):
        return _ref_formula(f.left, trace, i) and _ref_formula(f.right, trace, i)
    if isinstance(f, Or):
        return _ref_formula(f.left, trace, i) or _ref_formula(f.right, trace, i)
    if isinstance(f, Y):
        return i > 0 and _ref_formula(f.child, trace, i - 1)
    if isinstance(f, O):
        return any(_ref_formula(f.child, trace, j) for j in range(i + 1))
    if isinstance(f, H):
        return all(_ref_formula(f.child, trace, j) for j in range(i + 1))
    if isinstance(f, S):
        return any(
            _ref_formula(f.right, trace, j)
            and all(_ref_formula(f.left, trace, k) for k in range(j + 1, i + 1))
            for j in range(i + 1)
        )
    raise TypeError(f)


PAST_SHAPES = [
    Y(p),
    Y(Y(p)),
    O(p),
    H(p),
    S(p, q),
    S(Not(p), q),
    Y(S(p, q)),
    O(And(p, Y(q))),
    H(Or(p, O(q))),
    S(Or(p, r), And(q, Not(Y(p)))),
]


class TestPastFormulas:
    @pytest.mark.parametrize("f", PAST_SHAPES, ids=repr)
    def test_tracker_matches_definition(self, f):
        rng = random.Random(7)
        for _ in range(60):
            trace = _rand_trace(rng, rng.randint(1, 9))
            tracker = PastTracker(f)
            for i, ev in enumerate(trace):
                state = tracker.step(ev)
                assert state["__past__"][f] == _ref_formula(f, trace, i), (f, i)

    def test_y_is_false_at_the_first_event(self):
        assert evaluate(Y(p), [_ev(p_=True)]) is False
        assert evaluate(X(Y(p)), [_ev(p_=True), _ev()]) is True

    def test_h_and_o_at_the_first_event(self):
        assert evaluate(H(p), [_ev(p_=True)]) is True
        assert evaluate(O(p), [_ev()]) is False

    def test_position_reads_the_past_before_it(self):
        # At position 2, Y(p) reads event 1.
        trace = [_ev(), _ev(p_=True), _ev()]
        assert evaluate(Y(p), trace, pos=2) is True
        assert evaluate(Y(p), trace, pos=1) is False

    def test_precedence_with_since(self):
        # "every call was preceded by an approval, with no revoke in between"
        f = G(Implies(p, S(Not(r), q)))
        ok = [_ev(q_=True), _ev(), _ev(p_=True)]
        revoked = [_ev(q_=True), _ev(r_=True), _ev(p_=True)]
        assert evaluate(f, ok) and _dfa_holds(f, ok)
        assert not evaluate(f, revoked) and not _dfa_holds(f, revoked)

    def test_future_inside_past_is_refused(self):
        with pytest.raises(ValueError):
            past_nodes(O(F(p)))
        with pytest.raises(ValueError):
            evaluate(Y(G(p)), [_ev()])


# ---------------------------------------------------------------------------
# Past-time terms
# ---------------------------------------------------------------------------


class TestPastTerms:
    def test_prev(self):
        trace = [_ev(price_=100), _ev(price_=103), _ev(price_=99)]
        t1, t2 = Prev(price), Prev(price, 2)
        tr = PastTracker(Eq(Add(t1, t2), Const(0)))
        got = [tr.step(ev)["__past__"] for ev in trace]
        assert [g[t1] for g in got] == [None, 100, 103]
        assert [g[t2] for g in got] == [None, None, 100]

    def test_prev_rejects_bad_n(self):
        with pytest.raises(ValueError):
            Prev(price, 0)

    def test_change_since_last_step(self):
        # |price - prev(price)| <= 5, read only where there is a previous price
        move = Sub(price, Prev(price))
        f = G(And(Le(move, Const(5)), Ge(move, Const(-5))))
        calm = [_ev(price_=100), _ev(price_=103), _ev(price_=99)]
        jump = [_ev(price_=100), _ev(price_=110)]
        # At the first event Prev has no value, so the comparisons are false;
        # guard it the way a contract would, with Y(true)-style "has a past".
        guarded = G(Implies(O(Y(Not(Atom("called", "__never__")))), f.child))
        assert evaluate(guarded, calm) and _dfa_holds(guarded, calm)
        assert not evaluate(guarded, jump) and not _dfa_holds(guarded, jump)

    def test_at_freezes_the_value(self):
        # the value of price when p last held; react when price falls below 95% of it
        when_p = At(price, p)
        trace = [_ev(price_=100), _ev(p_=True, price_=100), _ev(price_=97), _ev(price_=94)]
        tr = PastTracker(Eq(when_p, Const(0)))
        assert [tr.step(ev)["__past__"][when_p] for ev in trace] == [None, 100, 100, 100]
        react = G(Implies(Lt(price, Mul(Const(0.95), when_p)), q))
        assert not evaluate(react, trace)  # 94 < 95 and no q
        trace[-1]["called(q)"] = True
        assert evaluate(react, trace) and _dfa_holds(react, trace)

    def test_at_updates_when_the_condition_holds_again(self):
        when_p = At(price, p)
        trace = [_ev(p_=True, price_=100), _ev(price_=90), _ev(p_=True, price_=80)]
        tr = PastTracker(Eq(when_p, Const(0)))
        assert [tr.step(ev)["__past__"][when_p] for ev in trace] == [100, 100, 80]

    def test_count_since(self):
        # number of ticks with r since the last p
        c = CountSince(r, p)
        trace = [_ev(r_=True), _ev(p_=True, r_=True), _ev(r_=True), _ev(), _ev(r_=True)]
        tr = PastTracker(Eq(c, Const(0)))
        assert [tr.step(ev)["__past__"][c] for ev in trace] == [None, 0, 1, 1, 2]

    def test_bounded_response_in_ticks(self):
        # after p, q within 2 more ticks: never 2 ticks past p without q
        answered = S(Not(p), q)  # q has held since the last p
        f = G(Not(And(Ge(CountSince(Atom("tick"), p), Const(2)), Not(answered))))
        ok = [_ev(p_=True), _ev(), _ev(q_=True), _ev()]
        late = [_ev(p_=True), _ev(), _ev(), _ev(q_=True)]
        for ev in ok + late:
            ev["tick()"] = True
        assert evaluate(f, ok) and _dfa_holds(f, ok)
        assert not evaluate(f, late) and not _dfa_holds(f, late)

    def test_term_formulas_are_annotated(self):
        # At's condition is itself a past formula
        t = At(price, Y(p))
        trace = [_ev(p_=True, price_=1), _ev(price_=2), _ev(price_=3)]
        out = annotate(Eq(t, Const(2)), trace)
        assert [ev["__past__"][t] for ev in out] == [None, 2, 2]


# ---------------------------------------------------------------------------
# The two evaluators agree
# ---------------------------------------------------------------------------


DIFF_SHAPES = [
    G(Implies(p, Y(q))),
    G(Implies(p, S(Not(r), q))),
    F(And(p, O(q))),
    G(Implies(p, H(Not(r)))),
    G(Or(Not(p), O(q))),
    G(Le(Add(price, qty), Const(110))),
    G(Implies(p, Le(price, Add(Prev(price), Const(5))))),
    G(Implies(Lt(price, Mul(Const(0.95), At(price, q))), r)),
    G(Le(CountSince(r, p), Const(2))),
    F(Ge(CountSince(r, q), Const(1))),
    G(Implies(p, X(Y(p)))),
    And(G(Implies(q, F(p))), G(Implies(p, O(q)))),
]


class TestDifferential:
    @pytest.mark.parametrize("f", DIFF_SHAPES, ids=repr)
    def test_recursive_and_dfa_agree(self, f):
        rng = random.Random(11)
        for _ in range(120):
            trace = _rand_trace(rng, rng.randint(1, 8))
            assert evaluate_final(f, trace) == _dfa_holds(f, trace), (f, trace)

    def test_snapshot_restore_keeps_the_past(self):
        f = G(Implies(p, Y(q)))
        dfa = DFAEvaluator(f)
        dfa.step(_ev(q_=True))
        snap = dfa.snapshot()
        assert dfa.step(_ev(p_=True)) != "⊥"
        dfa.restore(snap)
        assert dfa.step(_ev(p_=True)) != "⊥"
        dfa.reset()
        assert dfa.step(_ev(p_=True)) == "⊥"

    def test_annotation_extends_a_growing_trace(self):
        f = G(Implies(p, Y(q)))
        trace = [_ev(q_=True)]
        assert evaluate(f, trace)
        trace.append(_ev(p_=True))
        assert evaluate(f, trace)
        trace.append(_ev(p_=True))
        assert not evaluate(f, trace)


# ---------------------------------------------------------------------------
# Parser, text, satisfiability, verifier
# ---------------------------------------------------------------------------


ROUND_TRIP = [
    Y(p),
    S(Not(r), q),
    O(p),
    H(Not(r)),
    G(Implies(p, S(Not(r), q))),
    G(Le(Mul(price, qty), Const(1000))),
    G(Le(Sub(price, Prev(price, 2)), Const(5))),
    G(Implies(Lt(price, Mul(Const(0.95), At(price, q))), r)),
    G(Le(CountSince(r, p), Const(3))),
    G(Le(Div(Add(price, qty), Const(2)), Const(50))),
]


class TestParseAndText:
    @pytest.mark.parametrize("f", ROUND_TRIP, ids=repr)
    def test_parse_repr_round_trip(self, f):
        assert parse_repr(repr(f)) == f

    @pytest.mark.parametrize(
        ("text", "f"),
        [
            ("Y(called(p))", Y(p)),
            ("S(Not(called(r)), called(q))", S(Not(r), q)),
            ("G(Implies(called(p), O(called(q))))", G(Implies(p, O(q)))),
            ("H(Not(called(r)))", H(Not(r))),
            ("Le(Mul(Var(price), Var(qty)), Const(1000))", Le(Mul(price, qty), Const(1000))),
            (
                "Le(Sub(Var(price), Prev(Var(price), 2)), Const(5))",
                Le(Sub(price, Prev(price, 2)), Const(5)),
            ),
            (
                "Lt(Var(price), Mul(Const(0.95), At(Var(price), called(q))))",
                Lt(price, Mul(Const(0.95), At(price, q))),
            ),
            ("Le(CountSince(called(r), called(p)), Const(3))", Le(CountSince(r, p), Const(3))),
        ],
    )
    def test_parse_formula_constructors(self, text, f):
        assert parse_formula(text) == f

    def test_bare_atom_named_like_an_operator(self):
        # A bare name that is not followed by "(" stays an atom.
        assert parse_repr("S") != S

    @pytest.mark.parametrize("f", ROUND_TRIP, ids=repr)
    def test_text(self, f):
        assert isinstance(formula_to_nl(f), str) and formula_to_nl(f)


class TestSat:
    def test_past_nodes_do_not_create_false_conflicts(self):
        from contragent.formulas.sat import is_satisfiable

        assert is_satisfiable(G(Implies(p, Y(q))))
        assert is_satisfiable(And(G(Implies(p, S(Not(r), q))), F(p)))

    def test_plain_conflict_still_found(self):
        from contragent.formulas.sat import is_satisfiable

        assert not is_satisfiable(And(G(Not(p)), F(p)))


class TestVerifier:
    """Contracts written as ``ltl:`` strings, checked on real traces by both backends."""

    @staticmethod
    def _trace(*tools: str):
        from contragent.models.trace import Event, Trace

        return Trace(
            events=[
                Event(ts=i, agent="bot", event_type="tool_call", tool=t)
                for i, t in enumerate(tools)
            ]
        )

    @staticmethod
    def _both(formula, trace):
        from contragent.runtime.verifier import TraceVerifier

        out = []
        for backend in ("recursive", "dfa"):
            v = TraceVerifier(backend=backend)
            v.sync(trace)
            out.append(v.check(formula).holds)
        return out

    @pytest.mark.parametrize(
        ("tools", "expected"),
        [
            (("approve", "refund"), True),
            (("refund",), False),
            (("approve", "revoke", "refund"), False),
            (("approve", "revoke", "approve", "refund"), True),
        ],
    )
    def test_since_contract(self, tools, expected):
        from tests._helpers import ltl

        f = ltl("G((called('refund') -> S(!(called('revoke')), called('approve'))))")
        assert self._both(f, self._trace(*tools)) == [expected, expected]

    def test_incremental_sync(self):
        from contragent.runtime.verifier import TraceVerifier
        from tests._helpers import ltl

        f = ltl("G((called('refund') -> Y(called('lookup'))))")
        for backend in ("recursive", "dfa"):
            v = TraceVerifier(backend=backend)
            tools = ["lookup", "refund", "refund"]
            results = []
            for k in range(1, len(tools) + 1):
                v.sync(self._trace(*tools[:k]))
                results.append(v.check(f).holds)
            assert results == [True, True, False], backend


# ---------------------------------------------------------------------------
# Review fixes: walks that must see inside the new terms, the end of session
# ---------------------------------------------------------------------------


class TestWalksSeeInsideTerms:
    def test_numeric_argument_readers_are_found(self):
        """A call missing an argument a contract reads is refused, in or out of arithmetic."""
        from tests.test_fail_closed_gate import _guard

        plain = "G((called('refund') -> (Var('arg_numeric', 'refund', 'amount') <= 1000)))"
        total = (
            "G((called('refund') -> (Mul(Var('arg_numeric', 'refund', 'count'), "
            "Var('arg_numeric', 'refund', 'amount')) <= 1000)))"
        )
        for text in (plain, total):
            guard = _guard(text)
            assert guard.guard_before("refund", {}).blocked, text
            assert guard.guard_before("refund", {"count": 2, "amount": 100}).allowed, text
        assert not _guard(total).guard_before("refund", {"count": 20, "amount": 100}).allowed

    def test_content_atoms_inside_terms_are_grounded(self):
        from contragent.formulas.formula import Atom as A
        from contragent.tracer.grounding import collect_content_atoms

        when = A("called_with", "approve", "^big$")
        f = G(Le(Var("x"), At(Var("x"), when)))
        g = G(Le(CountSince(A("arg_field_has", "pay", "to", "^ext$"), p), Const(1)))
        found = collect_content_atoms([f, g])
        assert ("approve", "^big$") in found["called_with"]
        assert ("pay", "to", "^ext$") in found["arg_field_has"]

    def test_cumulative_vars_inside_terms_are_found(self):
        from contragent.runtime.verifier import _collect_cumulative_var_keys

        f = G(Le(Sub(Var("count", "pay"), Prev(Var("count", "pay"))), Const(1)))
        assert _collect_cumulative_var_keys(f) == {Var("count", "pay").key()}

    def test_foreach_inside_past_is_refused(self):
        from contragent.formulas.formula import ForEach

        with pytest.raises(ValueError):
            past_nodes(O(ForEach((("pay", "to"),), p)))


class TestEndOfSession:
    """After the last event the idle events' past values can still change."""

    def test_y_on_the_first_idle_event(self):
        # q at the last event: Y(q) holds on the first idle event only.
        for f, expected in [
            (F(Y(q)), True),
            (X(Y(q)), False),  # read at event 1: q did not hold at event 0
            (X(X(Y(q))), True),  # read at the first idle event
            (F(And(Not(Y(q)), O(q))), True),
            (U(Not(r), Y(q)), True),
        ]:
            trace = [_ev(), _ev(q_=True)]
            assert evaluate_final(f, trace) is expected, f
            assert _dfa_holds(f, trace) is expected, f

    def test_two_step_window_settles(self):
        trace = [_ev(p_=True)]
        tracker = PastTracker(Y(Y(p)))
        tracker.step(trace[0])
        idles = tracker.idle_events({"called(p)": False})
        assert [ev["__past__"][Y(Y(p))] for ev in idles] == [False, True, False]

    def test_g_binds_the_session_only(self):
        # G(Y(q)) fails at the first real event already; G is not read on idle events.
        assert evaluate_final(G(Implies(p, Y(q))), [_ev(q_=True), _ev(p_=True)]) is True


class TestAnnotationCache:
    def test_prefixes_and_growth_share_one_entry(self):
        f = G(Implies(p, Y(q)))
        base = [_ev(q_=True), _ev(p_=True), _ev(p_=True), _ev()]
        assert evaluate(f, base[:2]) is True
        assert evaluate(f, base) is False
        assert evaluate(f, base[:2]) is True  # a prefix of a longer annotation
        assert evaluate(f, base[:3]) is False

    def test_a_changed_last_event_starts_over(self):
        f = G(Implies(p, Y(q)))
        trace = [_ev(q_=True), _ev(p_=True)]
        assert evaluate(f, trace) is True
        trace[0]["called(q)"] = False
        trace[1]["called(p)"] = False
        assert evaluate(f, trace) is True
        trace[1]["called(p)"] = True
        assert evaluate(f, trace) is False

    def test_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        f = G(Implies(p, S(Not(r), q)))
        rng = random.Random(3)
        traces = [_rand_trace(rng, 30) for _ in range(8)]

        def run(i):
            t = traces[i % 8]
            return [evaluate(f, t[:k]) for k in range(1, 31)]

        expected = [run(i) for i in range(8)]
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(run, range(64)))
        assert got == [expected[i % 8] for i in range(64)]
