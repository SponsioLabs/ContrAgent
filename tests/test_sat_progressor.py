"""Tests for the public progression wrapper in formulas/sat.py."""

from contragent.formulas.formula import Atom, F, G, Implies
from contragent.formulas.sat import Progressor


def test_progressor_steps_and_finalizes():
    a, b = Atom("called", "a"), Atom("b")
    p = Progressor(G(Implies(a, F(b))))
    assert set(p.keys) == {"called(a)", "b()"}
    assert p.accepting(p.root)
    pending = p.step(p.root, {"called(a)"})
    assert not p.accepting(pending)
    assert p.accepting(p.step(pending, {"b()"}))
    assert p.step(p.root, set()) == p.root


def test_progressor_reaches_false_sink():
    a = Atom("a")
    p = Progressor(G(a))
    assert p.step(p.root, set()) is False
    assert p.step(False, {"a()"}) is False
