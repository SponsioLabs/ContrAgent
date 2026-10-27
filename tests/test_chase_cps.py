"""Tests for the chase_cps-based CHASE bridge (skipped without chase_cps or nuXmv)."""

from __future__ import annotations

import pytest

from contragent.analysis import check_conflicts
from contragent.analysis.chase import (
    ChaseSession,
    ChaseUnavailable,
    export_library,
    export_logics,
    is_available,
)
from contragent.config import bundled_libraries_root, load_system
from contragent.formulas.parser import parse_repr
from contragent.models.agent import Agent
from contragent.models.contract import Contract

SOPBENCH = sorted((bundled_libraries_root() / "sopbench").glob("*.yaml"))


def _nuxmv_present() -> bool:
    if not is_available():
        return False
    from chase_cps.backends.nusmv import NuSMVAdapter

    return NuSMVAdapter(tool="nuxmv").available()


needs_chase = pytest.mark.skipif(
    not _nuxmv_present(), reason="requires chase_cps (CHASE's Python package) and nuXmv"
)


def _relations(node, out):
    """``(variable, constant)`` for every relation between a variable and a constant."""
    from chase_cps import ltl

    if isinstance(node, ltl.Binary):
        if node.op in ltl.RELATIONS:
            for var, num in ((node.left, node.right), (node.right, node.left)):
                if isinstance(var, ltl.Var) and isinstance(num, ltl.Num):
                    out.append((var.name, num.value))
        _relations(node.left, out)
        _relations(node.right, out)
    elif isinstance(node, ltl.Unary):
        _relations(node.arg, out)
    return out


def _contracts(*guarantees):
    bot = Agent(id="bot")
    return [
        Contract(agent=bot, guarantee=parse_repr(g), desc=f"rule {i}")
        for i, g in enumerate(guarantees)
    ]


def test_session_reports_missing_package_clearly():
    if is_available():
        pytest.skip("chase_cps is installed")
    with pytest.raises(ChaseUnavailable):
        ChaseSession(export_library(SOPBENCH[0]))


@needs_chase
@pytest.mark.parametrize("path", SOPBENCH, ids=lambda p: p.stem)
def test_every_sopbench_contract_verifies(path):
    session = ChaseSession(export_library(path))
    assert len(session.contracts) == len(load_system(path).contracts)
    for ident in session.contracts:
        result = session.verify(ident)
        assert result, (path.name, ident, result.to_dict())


@needs_chase
def test_university_ranges_contain_the_compared_constants():
    import chase_cps

    contracts = chase_cps.parse_spec(
        export_library(bundled_libraries_root() / "sopbench" / "university.yaml")
    )
    compared = set()
    for c in contracts.values():
        for term in c.assumptions + c.guarantees:
            for name, value in _relations(term, []):
                lo, hi = c.variables[name]
                assert lo <= value <= hi, (name, value, (lo, hi))
                compared.add(value)
    assert 90 in compared


@needs_chase
def test_unsat_core_names_an_injected_conflict():
    system = load_system(bundled_libraries_root() / "sopbench" / "university.yaml")
    agent = system.contracts[0].agent
    clash = [
        Contract(
            agent=agent,
            guarantee=parse_repr("F(called('audit_transcript'))"),
            desc="the transcript is audited",
        ),
        Contract(
            agent=agent,
            guarantee=parse_repr("G(!(called('audit_transcript')))"),
            desc="the transcript is never audited",
        ),
    ]
    library = list(system.contracts)
    session = ChaseSession.from_library(library + clash, name="university")
    pair = [f"c{len(library) + 1}", f"c{len(library) + 2}"]
    core = session.unsat_core()
    assert core, core.to_dict()
    # The finite-trace axiom may take part: F(audit) could otherwise be met
    # after the end of the session, where no tool is called.
    assert all(t in session.axiom for t in core.core.assumptions)
    assert session.contracts_in(core) == pair

    # The native search cannot settle the whole unsatisfiable library (its
    # alphabet grows with every comparison), so on the whole library the
    # check agrees on what makes the pair the only core: the library is
    # conflict-free with either contract of the pair.
    for c in clash:
        assert check_conflicts(library + [c]).ok, c.desc

    # On a part of the library small enough for the native search, with the
    # comparisons with 90, both checks name the pair and nothing else.
    part = [i for i, c in enumerate(library) if "90" in repr(c.guarantee)] + [0, 1]
    assert len(part) == 4
    report = check_conflicts([library[i] for i in part] + clash)
    assert not report.unknown
    assert [set(c.labels) for c in report.conflicts] == [{c.desc for c in clash}]
    idents = [f"c{i + 1}" for i in part] + pair
    assert session.contracts_in(session.unsat_core(idents), idents) == pair


@needs_chase
def test_consistency_is_read_on_finite_traces():
    import chase_cps

    # G(a) & G(!a) holds on the words on which alive is false from the
    # start, so CHASE's own consistency check passes it; on finite traces,
    # which the axiom keeps, it fails.
    session = ChaseSession.from_library(
        _contracts("G(called('a'))", "(G(called('a')) & G(!(called('a'))))")
    )
    assert session.verify("c1")
    assert chase_cps.verify(session.contracts["c2"])
    result = session.verify("c2")
    assert result.status == "unsat" and result.compatible


@needs_chase
def test_refinement_and_redundancy():
    session = ChaseSession(
        export_logics(
            _contracts(
                "G(!(called('a')))",
                "(G(!(called('a'))) & G((called('b') -> F(called('c')))))",
                "(G(!(called('a'))) & G((!(called('a')) | called('b'))))",
            )
        )
    )
    assert session.refines("c2", "c1")
    result = session.refines("c1", "c2")
    assert result.status == "invalid" and result.failed == "guarantees"
    assert result.guarantees.trace is not None
    assert len(session.remove_redundant("c2").contract.guarantees) == 2
    redundancy = session.remove_redundant("c3")
    assert len(redundancy.contract.guarantees) == 1 and len(redundancy.removed_guarantees) == 1
    assert not redundancy.contract.assumptions
    assert session.conjoin(["c1", "c2"], name="both").name == "both"


@needs_chase
@pytest.mark.parametrize(
    "guarantee",
    [
        "(G((called('m') -> X(called('g')))) & F(called('m')) & G(!(called('g'))))",
        "(G((called('m') -> X(called('g')))) & F(called('m')))",
        "(G((called('s') -> X(!(called('s'))))) & F(called('s')))",
        "(G((called('m') -> X(X(called('g'))))) & F(called('m')) & G(!(called('g'))))",
        "(X(ctx('auth', 'yes')) & G(!(ctx('auth', 'yes'))))",
        "(G(called('a')) & F(called('a')))",
    ],
)
def test_finite_export_agrees_with_the_satisfiability_check(guarantee):
    from contragent.formulas.sat import is_satisfiable

    (c,) = _contracts(guarantee)
    session = ChaseSession.from_library([c])
    assert bool(session.verify("c1")) is bool(is_satisfiable(c.guarantee)), guarantee
