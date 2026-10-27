"""Bridge to CHASE, the contract-based requirement-engineering framework.

CHASE (Nuzzo, Lora, Feldman, and Sangiovanni-Vincentelli, DATE 2018;
https://chase-cps.github.io) represents assume-guarantee contracts over
temporal-logic formulas and offers a contract algebra (composition,
conjunction, refinement) with model-checking and synthesis back ends
(NuSMV, slugs, gr1c). ContrAgent's contract libraries are ALTL\\ :sub:`f`
contracts over interaction predicates; this module maps them onto CHASE's
*logics* specification language so that a library can be analysed with
CHASE's design-time tools, and checks the exported contracts with
``chase_cps``, CHASE's Python package, when it is installed.

Grounding
---------
CHASE's logics language is propositional with typed variables: atoms are
propositions or relations over integer/real/boolean variables. ContrAgent's
predicates carry parameters, so a library is *grounded* at export:

* every distinct instantiated predicate (``called(transfer)``,
  ``arg_field_has(bash, command, rm -rf)``, ...) becomes one proposition,
  named by its canonical key (:func:`contragent.formulas._pred_key.pred_key`);
* every quantity (``count(T)``, ``arg_numeric(T, f)``, ``ArgLength(T, f)``,
  ``CtxValue(k)``) becomes one integer variable declared with a range;
* comparisons are emitted as relations inside the formulas.

The logics language gives an integer declared without a range the values
0..16, so a comparison with a larger constant (``university.yaml`` compares
credits with 90) would be checked on the wrong values; every integer is
therefore declared ``integer (lo:hi) variable``:

* a saturating counter gets ``(0:K)``, where ``K`` is one more than the
  largest constant it is compared with (at least 1), the same finite
  expansion the satisfiability check uses;
* any other integer gets ``(lo:hi)``, where ``lo`` and ``hi`` are the
  smallest and the largest constant it is compared with, widened by one;
  an ``arg_length`` never goes below 0;
* integers compared with each other share one range: the union of their
  constants, widened by the number of integers in the group instead of one,
  or ``(0:n-1)`` for ``n`` integers compared with no constant.

The ranges are exact. A comparison with a constant cannot tell apart two
values below the smallest constant ``c_lo``, nor two values above the
largest ``c_hi``, so ``c_lo - 1`` stands for every smaller value and
``c_hi + 1`` for every larger one, and the range keeps every integer in
between; ``K`` stands for every count above the constants. A group of
``n`` integers compared with each other takes at most ``n`` distinct values
below ``c_lo`` at one step, and the ``n`` places below it keep their order
(likewise above ``c_hi``). Relations compare values of the same step, so a
trace over the integers maps step by step onto a trace over the ranges
that satisfies the same relations, and a formula is satisfiable over the
declared ranges exactly when it is over the integers. The map sends a
value below ``c_lo`` to a place no smaller than the value, so a counter
and a length, which are never negative, keep the range from 0; ``ctx_value`` and
``arg_numeric`` carry any number the agent supplies, negative ones
included, so their range keeps the places below ``c_lo`` even when that
is below 0. ``arg_numeric`` is read on the integers: a real argument
strictly between two consecutive integer constants has no counterpart.

Finite traces
-------------
CHASE's back ends reason over infinite words while ContrAgent's monitors
evaluate LTL\\ :sub:`f` on finite traces. ContrAgent decides a finished
session by reading what it still owes on idle events after the last one:
no tool is called and the state stays what it was, while a ``G`` binds the
events of the session (:func:`contragent.formulas.evaluator.evaluate_final`).
With ``semantics="finite"`` (the default) the export writes that reading
with a fresh proposition ``alive`` that marks the events of the session:
``G`` becomes ``G(alive -> .)``, ``X``, ``F``, and ``U`` are kept, and a
contract ``finite`` asserts ``alive /\\ (alive U G(!alive))``, that every
event proposition (a call, its arguments, an LLM message) is false once
``alive`` is, and that every state proposition and every relation over
state quantities keeps its value from the last event of the session on.
A relation over the arguments of a call is left free after the end, so a
check over the export allows those idle values the runtime reads as
missing. With ``semantics="infinite"`` the formulas are exported unchanged.

Usage
-----
::

    from contragent.analysis.chase import export_logics
    from contragent.config import load_system

    system = load_system("contragent/contracts/sopbench/bank.yaml")
    text = export_logics(system.contracts, name="bank")
    open("bank.logics", "w").write(text)

    # with chase_cps (CHASE's Python package) and nuXmv installed:
    from contragent.analysis.chase import ChaseSession
    s = ChaseSession.from_file("bank.logics")
    s.verify("c1").status            # sat, or unsat; each part carries its trace
    s.refines("c1", "c2")            # valid, or invalid with a counterexample
    s.unsat_core()                   # the terms that cannot hold together

or ``contragent export-chase --config LIB -o LIB.logics``.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contragent.formulas._pred_key import EVENT_PREDICATES
from contragent.formulas.formula import (
    And,
    ArgLength,
    ArgValue,
    Atom,
    Const,
    CtxValue,
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
    Subset,
    U,
    UnaryFn,
    Var,
    X,
)

_SATURATING = frozenset(
    {"count", "count_with", "token_count", "consecutive_count", "delegation_depth", "time_since"}
)
_IDENT_RE = re.compile(r"[^A-Za-z0-9_]+")


class ChaseExportError(ValueError):
    """A formula uses a construct that has no CHASE counterpart."""


class ChaseUnavailable(RuntimeError):
    """CHASE's Python package (``chase_cps``) is not installed, or is too old."""


# CHASE's console (verify, refinement, synthesize) crashes on contract names
# longer than eight characters, so exported contracts are numbered and the
# description travels in the comment above each block.
MAX_CONTRACT_IDENT = 8
FINITE_TRACE_CONTRACT = "finite"


def _comment(text: str) -> str:
    """CHASE line comments run to the end of the line and may not contain backslashes."""
    return str(text).replace("\\", "").replace("\r", " ").replace("\n", " ")


def _ident(prefix: str, *parts: Any) -> str:
    raw = "_".join(str(p) for p in parts)
    raw = _IDENT_RE.sub("_", raw).strip("_")
    name = f"{prefix}_{raw}" if raw else prefix
    return name if name[0].isalpha() else f"p_{name}"


@dataclass
class _Symbols:
    """Declarations collected while walking a library."""

    propositions: dict[str, str] = field(default_factory=dict)  # ident -> comment
    integers: dict[str, str] = field(default_factory=dict)  # ident -> comment
    constants: dict[str, set[float]] = field(default_factory=dict)  # ident -> compared constants
    links: dict[str, set[str]] = field(default_factory=dict)  # ident -> compared integers
    saturating: set[str] = field(default_factory=set)
    nonnegative: set[str] = field(default_factory=set)  # lengths: never below 0
    event_propositions: set[str] = field(default_factory=set)  # false after the end
    state_relations: dict[str, None] = field(default_factory=dict)  # kept after the end

    def proposition(self, ident: str, comment: str) -> str:
        self.propositions.setdefault(ident, comment)
        return ident

    def integer(
        self, ident: str, comment: str, *, saturating: bool, nonnegative: bool = False
    ) -> str:
        self.integers.setdefault(ident, comment)
        if saturating:
            self.saturating.add(ident)
        if nonnegative:
            self.nonnegative.add(ident)
        return ident

    def note_constant(self, ident: str, value: float) -> None:
        self.constants.setdefault(ident, set()).add(float(value))

    def note_link(self, a: str, b: str) -> None:
        self.links.setdefault(a, set()).add(b)
        self.links.setdefault(b, set()).add(a)

    def _group(self, ident: str) -> list[str]:
        """The integers connected to ``ident`` by comparisons between variables."""
        group, todo = [ident], [ident]
        while todo:
            for other in sorted(self.links.get(todo.pop(), ())):
                if other not in group:
                    group.append(other)
                    todo.append(other)
        return group

    def ranges(self) -> dict[str, tuple[int, int]]:
        """The declared range of every integer; the module docstring says why it is exact."""
        out: dict[str, tuple[int, int]] = {}
        for ident in self.integers:
            if ident in out:
                continue
            group = self._group(ident)
            consts = set().union(*(self.constants.get(v, set()) for v in group))
            n = len(group)
            if consts:
                lo, hi = math.floor(min(consts)) - n, math.ceil(max(consts)) + n
            else:
                lo, hi = 0, n - 1
            if any(v in self.saturating or v in self.nonnegative for v in group):
                # n places from 0 up for the values of a group whose constants are all negative.
                hi = max(hi, n - 1)
            for v in group:
                if v in self.saturating:
                    # A counter is never negative; K = hi stands for every larger count.
                    out[v] = (0, max(hi, 1))
                elif v in self.nonnegative:
                    out[v] = (max(lo, 0), hi)
                else:
                    out[v] = (lo, hi)
        return out


def _term(term: Any, sym: _Symbols) -> tuple[str, str | None]:
    """Return ``(text, variable_ident_or_None)`` for a numeric term."""
    if isinstance(term, Const):
        v = term.value
        return (str(int(v)) if float(v).is_integer() else str(v)), None
    if isinstance(term, Var):
        # Quantities written in the infix syntax arrive as ``Var`` nodes named
        # after the quantity; give them the same identifiers as the typed terms.
        if term.name in ("arg_value", "arg_numeric", "ArgValue") and len(term.args) == 2:
            ident = _ident("num", *term.args)
            return sym.integer(
                ident, f"arg_numeric({term.args[0]}, {term.args[1]})", saturating=False
            ), ident
        if term.name in ("arg_length", "ArgLength") and len(term.args) == 2:
            ident = _ident("len", *term.args)
            return sym.integer(
                ident,
                f"arg_length({term.args[0]}, {term.args[1]})",
                saturating=False,
                nonnegative=True,
            ), ident
        if term.name in ("ctx_value", "CtxValue") and len(term.args) == 1:
            ident = _ident("ctx", term.args[0])
            return sym.integer(ident, f"ctx_value({term.args[0]})", saturating=False), ident
        ident = _ident("v", term.name, *term.args)
        return sym.integer(ident, term.key(), saturating=term.name in _SATURATING), ident
    if isinstance(term, ArgValue):
        ident = _ident("num", term.tool, term.field)
        return sym.integer(
            ident, f"arg_numeric({term.tool}, {term.field})", saturating=False
        ), ident
    if isinstance(term, ArgLength):
        ident = _ident("len", term.tool, term.field)
        return sym.integer(
            ident, f"arg_length({term.tool}, {term.field})", saturating=False, nonnegative=True
        ), ident
    if isinstance(term, CtxValue):
        ident = _ident("ctx", term.key)
        return sym.integer(ident, f"ctx_value({term.key})", saturating=False), ident
    if isinstance(term, UnaryFn):
        inner, _ = _term(term.arg, sym)
        ident = _ident("fn", term.name, inner)
        return sym.integer(ident, f"{term.name}({inner})", saturating=False), ident
    raise ChaseExportError(f"unsupported numeric term {type(term).__name__}")


def _relation(node: Any, op: str, sym: _Symbols) -> str:
    from contragent.formulas.sat import _term_is_event

    left, lvar = _term(node.left, sym)
    right, rvar = _term(node.right, sym)
    if lvar and isinstance(node.right, Const):
        sym.note_constant(lvar, node.right.value)
    if rvar and isinstance(node.left, Const):
        sym.note_constant(rvar, node.left.value)
    if lvar and rvar:
        sym.note_link(lvar, rvar)
    text = f"({left} {op} {right})"
    if True not in (_term_is_event(node.left), _term_is_event(node.right)):
        sym.state_relations.setdefault(text, None)
    return text


def _formula(node: Any, sym: _Symbols, finite: bool) -> str:
    """Translate one formula node into CHASE logics syntax."""
    if isinstance(node, Atom):
        ident = _ident("p", node.predicate, *node.args)
        if node.predicate in EVENT_PREDICATES:
            sym.event_propositions.add(ident)
        return sym.proposition(ident, node.key())
    if isinstance(node, Subset):
        ident = _ident("subset", node.left, node.right)
        return sym.proposition(ident, f"subset({node.left}, {node.right})")
    if isinstance(node, Not):
        return f"(! {_formula(node.child, sym, finite)})"
    if isinstance(node, And):
        return f"({_formula(node.left, sym, finite)} /\\ {_formula(node.right, sym, finite)})"
    if isinstance(node, Or):
        return f"({_formula(node.left, sym, finite)} \\/ {_formula(node.right, sym, finite)})"
    if isinstance(node, Implies):
        return f"({_formula(node.left, sym, finite)} -> {_formula(node.right, sym, finite)})"
    if isinstance(node, G):
        body = _formula(node.child, sym, finite)
        return f"(G (alive -> {body}))" if finite else f"(G {body})"
    if isinstance(node, F):
        return f"(F {_formula(node.child, sym, finite)})"
    if isinstance(node, X):
        return f"(X {_formula(node.child, sym, finite)})"
    if isinstance(node, U):
        return f"({_formula(node.left, sym, finite)} U {_formula(node.right, sym, finite)})"
    if isinstance(node, Le):
        return _relation(node, "<=", sym)
    if isinstance(node, Lt):
        return _relation(node, "<", sym)
    if isinstance(node, Ge):
        return _relation(node, ">=", sym)
    if isinstance(node, Gt):
        return _relation(node, ">", sym)
    if isinstance(node, Eq):
        return _relation(node, "=", sym)
    raise ChaseExportError(f"unsupported formula node {type(node).__name__}")


def _raw(constraint: Any) -> Any:
    return getattr(constraint, "formula", constraint)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


@dataclass
class ExportedContract:
    ident: str
    desc: str
    assumptions: list[str]
    guarantees: list[str]


def export_logics(
    contracts: Sequence[Any],
    *,
    name: str = "contragent_library",
    semantics: str = "finite",
) -> str:
    """Render a contract library as a CHASE logics specification.

    Args:
        contracts: :class:`~contragent.models.contract.Contract` objects (or
            anything with ``assumptions`` / ``guarantees`` lists of
            formulas or ``DetFormula`` wrappers).
        name: The ``NAME:`` of the CHASE system.
        semantics: ``"finite"`` applies the LTL\\ :sub:`f`-to-LTL
            translation with the ``alive`` proposition; ``"infinite"``
            exports the formulas unchanged.
    """
    if semantics not in ("finite", "infinite"):
        raise ValueError("semantics must be 'finite' or 'infinite'")
    finite = semantics == "finite"
    sym = _Symbols()
    if finite:
        sym.proposition("alive", "the finite trace has not ended")
    exported: list[ExportedContract] = []
    for i, c in enumerate(contracts, start=1):
        desc = getattr(c, "desc", None) or f"contract {i}"
        ident = f"c{i}"
        assert len(ident) <= MAX_CONTRACT_IDENT
        a = [_formula(_raw(x), sym, finite) for x in _as_list(getattr(c, "assumptions", []))]
        g = [_formula(_raw(x), sym, finite) for x in _as_list(getattr(c, "guarantees", []))]
        exported.append(ExportedContract(ident, desc, a, g))

    out: list[str] = []
    out.append(f"# CHASE logics specification exported by ContrAgent ({semantics}-trace semantics)")
    system_name = _IDENT_RE.sub("_", name).strip("_") or "contragent_library"
    if not system_name[0].isalpha():
        system_name = f"lib_{system_name}"
    out.append(f"NAME: {system_name};")
    out.append("")
    out.append("# --- propositions: one per grounded interaction predicate ---")
    for ident, comment in sym.propositions.items():
        out.append(f"proposition {ident};  # {_comment(comment)}")
    if sym.integers:
        out.append("")
        out.append("# --- quantities: integers over the range their comparisons tell apart ---")
        ranges = sym.ranges()
        for ident, comment in sym.integers.items():
            lo, hi = ranges[ident]
            note = f", saturates at {hi}" if ident in sym.saturating else ""
            out.append(f"integer ({lo}:{hi}) variable {ident};  # {_comment(comment)}{note}")
    out.append("")
    if finite:
        out.append("# Finite-trace axiom: a non-empty session, then idle events forever, at which")
        out.append("# no event predicate holds and the state stays what it was at the last event.")
        out.append(f"CONTRACT {FINITE_TRACE_CONTRACT}:")
        out.append("  Assumptions:")
        out.append("    true;")
        out.append("  Guarantees:")
        out.append("    (alive /\\ (alive U (G (! alive))));")
        events = [p for p in sym.propositions if p in sym.event_propositions]
        if events:
            none = " /\\ ".join(f"(! {p})" for p in events)
            out.append(f"    (G ((! alive) -> ({none})));")
        kept = [
            p for p in sym.propositions if p != "alive" and p not in sym.event_propositions
        ] + list(sym.state_relations)
        if kept:
            same = " /\\ ".join(f"({k} <-> (X {k}))" for k in kept)
            out.append(f"    (G ((X (! alive)) -> ({same})));")
        out.append("")
    for ec in exported:
        out.append(f"# {_comment(ec.desc)}")
        out.append(f"CONTRACT {ec.ident}:")
        # CHASE's console expects both blocks; an unconditional contract assumes true.
        out.append("  Assumptions:")
        for a in ec.assumptions or ["true"]:
            out.append(f"    {a};")
        out.append("  Guarantees:")
        for g in ec.guarantees or ["true"]:
            out.append(f"    {g};")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def export_library(
    config_path: str | Path,
    out_path: str | Path | None = None,
    *,
    agent_id: str | None = None,
    semantics: str = "finite",
) -> str:
    """Export a library file to CHASE logics syntax and optionally write it."""
    from contragent.config import load_system

    system = load_system(config_path)
    contracts = list(system.contracts)
    if agent_id:
        contracts = [c for c in contracts if c.agent.id == agent_id]
    text = export_logics(contracts, name=Path(config_path).stem, semantics=semantics)
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


# ----------------------------------------------------------------------
# Optional: check an exported library with chase_cps
# ----------------------------------------------------------------------


def is_available() -> bool:
    """True when ``chase_cps`` and the CHASE modules it is built on can be
    imported, and its checks take the ``context=`` the session passes."""
    try:
        import chase_cps  # type: ignore[import-not-found]

        # chase_cps imports without its compiled modules and fails at the first
        # check; _native is where they load, until chase_cps has a public check.
        import chase_cps._native  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return "context" in inspect.signature(chase_cps.verify).parameters


def unavailable_reason() -> str:
    return (
        "chase_cps, the Python package of CHASE, is not installed, or does not "
        "take the context= of a check: pip install 'chase-cps>=0.3.0,<0.4' "
        "(or 'contragent[chase]'). The checks run nuXmv; `chase-cps doctor` shows "
        "the tools found."
    )


class ChaseSession:
    """The contracts of an exported specification, read by ``chase_cps``.

    ``contracts`` maps each contract identifier (``c1``, ``c2``, ...) to a
    ``chase_cps.Contract`` over the variables the contract uses. The
    finite-trace axiom of a ``semantics="finite"`` export is kept apart in
    ``axiom`` and passed as the ``context`` of every check below, which
    reads the contracts on the traces it allows. The checks run nuXmv through ``chase_cps`` and return its
    results, with a status, the counterexample or witness trace, and the
    core. Requires ``chase_cps``; see :func:`unavailable_reason`.
    """

    def __init__(self, text: str) -> None:
        if not is_available():
            raise ChaseUnavailable(unavailable_reason())
        import chase_cps  # type: ignore[import-not-found]

        self._cps = chase_cps
        self.text = text
        parsed = chase_cps.parse_spec(text)
        finite = parsed.pop(FINITE_TRACE_CONTRACT, None)
        self.axiom: list = list(finite.guarantees) if finite is not None else []
        self._axiom_vars: dict = dict(finite.variables) if finite is not None else {}
        self.contracts: dict[str, Any] = {
            ident: parsed[ident] for ident in contract_identifiers(text) if ident in parsed
        }
        # chase-cps before 0.2.0 leaves the declarations at the top of a
        # specification on the system, so its contracts come back without variables.
        for c in self.contracts.values():
            used = chase_cps.ltl.variables(chase_cps.ltl.conjunction(c.assumptions + c.guarantees))
            if not used <= set(c.variables):
                raise ChaseUnavailable(
                    "this chase_cps does not give contracts the declarations shared at "
                    "the top of a specification. " + unavailable_reason()
                )

    @classmethod
    def from_file(cls, logics_path: str | Path) -> ChaseSession:
        """A session on an exported ``.logics`` file."""
        return cls(Path(logics_path).read_text())

    @classmethod
    def from_library(
        cls,
        contracts: Sequence[Any],
        *,
        name: str = "contragent_library",
        semantics: str = "finite",
    ) -> ChaseSession:
        """A session on ``export_logics(contracts, ...)``."""
        return cls(export_logics(contracts, name=name, semantics=semantics))

    def _variables(self, contracts: Sequence[Any]) -> dict:
        variables = dict(self._axiom_vars) if self.axiom else {}
        for c in contracts:
            variables.update(c.variables)
        return variables

    def _in_context(self, c: Any) -> Any:
        """``c`` with the axiom's variables declared, so the axiom can be its context."""
        missing = {k: v for k, v in self._axiom_vars.items() if k not in c.variables}
        if not missing:
            return c
        return self._cps.Contract(
            c.name,
            dict(c.inputs),
            {**missing, **c.outputs},
            list(c.assumptions),
            list(c.guarantees),
        )

    def _context(self) -> dict:
        return {"context": self.axiom} if self.axiom else {}

    def verify(self, ident: str, **options: Any) -> Any:
        """Compatibility and consistency of a contract (``chase_cps.verify``).

        The finite-trace axiom is the ``context`` of the check: a word that
        breaks it is no behaviour at all, so compatibility is ``axiom & A``
        satisfiable and consistency ``axiom & (A -> G)`` satisfiable. Taken
        as an assumption instead, the axiom would let consistency hold on
        any word that breaks it.
        """
        c = self._in_context(self.contracts[ident])
        return self._cps.verify(c, **self._context(), **options)

    def refines(self, refined: str, abstract: str, **options: Any) -> Any:
        """Whether ``refined`` refines ``abstract`` (``chase_cps.refines``).

        The axiom is the context. Added to the assumptions of both contracts
        it gives the same verdict, since both refinement conditions are
        implications whose premise holds the assumptions, but the context
        keeps the contracts as they were exported and the axiom out of the
        terms a counterexample is read against.
        """
        r, a = self._in_context(self.contracts[refined]), self._in_context(self.contracts[abstract])
        return self._cps.refines(r, a, **self._context(), **options)

    def remove_redundant(self, ident: str, **options: Any) -> Any:
        """The contract without the terms implied by the others (``chase_cps.remove_redundant``).

        The axiom is the context: a term is redundant when the others imply
        it on finite traces, and the axiom, not being a term of the
        contract, is never removed or returned.
        """
        c = self._in_context(self.contracts[ident])
        return self._cps.remove_redundant(c, **self._context(), **options)

    def unsat_core(self, idents: Sequence[str] | None = None, **options: Any) -> Any:
        """A minimal set of terms of the contracts that cannot hold together.

        The terms are the assumptions and the guarantees of the contracts,
        the conjunction :func:`contragent.analysis.check_conflicts` checks.
        The finite-trace axiom is the context of every test, so it never
        appears in the core; ``chase_cps.unsat_core`` returns a contract
        with the terms of the core, and :meth:`contracts_in` names the
        contracts they come from.
        """
        chosen = [self.contracts[i] for i in (idents or self.contracts)]
        library = self._cps.Contract(
            "library",
            {},
            self._variables(chosen),
            [t for c in chosen for t in c.assumptions],
            [t for c in chosen for t in c.guarantees],
        )
        return self._cps.unsat_core(library, **self._context(), **options)

    def contracts_in(self, core: Any, idents: Sequence[str] | None = None) -> list[str]:
        """The identifiers of the contracts an :meth:`unsat_core` core comes from.

        ``idents`` are the contracts the core was searched in. A core holds
        terms, not their positions, so a term that several contracts share
        (in ``university.yaml`` seventeen pairs of contracts share a term)
        is put down to the first of them; a minimal core holds one copy of it.
        """
        chosen = list(idents or self.contracts)
        found = set()
        for t in list(core.core.assumptions) + list(core.core.guarantees):
            for ident in chosen:
                c = self.contracts[ident]
                if t in c.assumptions or t in c.guarantees:
                    found.add(ident)
                    break
        return [ident for ident in chosen if ident in found]

    def conjoin(self, idents: Sequence[str], name: str = "library") -> Any:
        """The CHASE conjunction of the contracts (``chase_cps.Contract.conjoin``).

        Conjunction is the operation for several contracts on the *same*
        component, which is what a library of contracts on one agent is.
        """
        acc = self.contracts[idents[0]]
        for ident in idents[1:]:
            acc = acc.conjoin(self.contracts[ident], name=name)
        return acc


def contract_identifiers(text: str) -> list[str]:
    """The ``CONTRACT`` identifiers declared in an exported specification."""
    return re.findall(r"^CONTRACT\s+([A-Za-z][A-Za-z0-9_]*):", text, flags=re.M)


_DECL_RE = re.compile(
    r"^(?P<kind>proposition|integer(?: \((?P<lo>-?\d+):(?P<hi>-?\d+)\))? variable)"
    r" (?P<name>[A-Za-z]\w*);",
    flags=re.M,
)


def smv_declarations(text: str, *, int_range: tuple[int, int] | None = None) -> list[str]:
    """NuSMV ``VAR`` lines for the declarations of a logics specification.

    The ``verify`` command of the standalone ``logics_tool`` writes only the
    variables attached to the contract, and its parser attaches the
    declarations to the system, so the model it emits for an exported
    specification has an empty ``VAR`` block. Propositions become booleans
    and ranged integers keep their range. An integer without a range (not
    written by :func:`export_logics`) gets ``int_range`` when one is given
    (nuXmv's BDD algorithms need finite domains) and the ``integer`` type
    otherwise (for its SMT-based algorithms, e.g. ``msat_check_ltlspec_bmc``).
    """
    lines: list[str] = []
    for m in _DECL_RE.finditer(text):
        if m["kind"] == "proposition":
            typ = "boolean"
        elif m["lo"] is not None:
            typ = f"{m['lo']}..{m['hi']}"
        elif int_range is not None:
            typ = f"{int_range[0]}..{int_range[1]}"
        else:
            typ = "integer"
        lines.append(f"\t{m['name']} : {typ};\n")
    return lines


__all__ = [
    "FINITE_TRACE_CONTRACT",
    "MAX_CONTRACT_IDENT",
    "ChaseExportError",
    "ChaseUnavailable",
    "ChaseSession",
    "export_logics",
    "export_library",
    "contract_identifiers",
    "smv_declarations",
    "is_available",
    "unavailable_reason",
]
