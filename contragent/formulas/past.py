"""Past-time operators: keeping what the history says, one event at a time.

A past-time node (``Y``, ``S``, ``O``, ``H`` and the terms ``Prev``, ``At``,
``CountSince``) is decided at an event by the events up to and including it.
:class:`PastTracker` keeps a few values per node and, for each event, puts
the value of every past-time node of a formula into the event's valuation
under :data:`~contragent.formulas.formula.PAST_KEY`. The evaluators then read
a past-time node like an atom, so the future-time semantics and the monitor's
progression are unchanged (the construction of Havelund and Rosu, "Synthesizing
monitors for safety properties", TACAS 2002).

The value at an event depends only on the events before and at it, never on
where in a formula the node sits or on later events, so one valuation per
event serves every occurrence of the node and every evaluator.

Semantics, at event ``i`` of a trace (the first event is ``0``):

* ``Y(p)``: ``i > 0`` and ``p`` held at ``i - 1``.
* ``S(p, q)``: ``q`` held at some ``j <= i`` and ``p`` at every ``k`` with ``j < k <= i``.
* ``O(p)``: ``p`` held at some ``j <= i``. ``H(p)``: at every ``j <= i``.
* ``Prev(t, n)``: the value of ``t`` at ``i - n``; ``None`` when ``i < n``.
* ``At(t, w)``: the value of ``t`` at the last ``j <= i`` where ``w`` held; ``None`` if none.
* ``CountSince(c, s)``: the number of ``k`` with ``j < k <= i`` where ``c`` held,
  ``j`` the last event ``<= i`` where ``s`` held; ``None`` if ``s`` never held.

Operands are pointwise or past-time formulas. A future operator (``G``, ``F``,
``X``, ``U``) inside a past-time node has no value at the current event and is
refused with ``ValueError``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from contragent.formulas.formula import (
    PAST_FORMULAS,
    PAST_KEY,
    PAST_TERMS,
    At,
    CountSince,
    F,
    ForEach,
    G,
    H,
    O,
    Prev,
    S,
    Term,
    U,
    X,
    Y,
    term_children,
)

__all__ = ["PastTracker", "annotate", "has_past", "past_nodes"]

_FUTURE = (G, F, X, U)


def _children(node: Any) -> list:
    """Formulas and terms directly under a node (formula or term)."""
    if isinstance(node, Term):
        return list(term_children(node)) + list(term_formulas_here(node))
    out: list = []
    for attr in ("child", "left", "right"):
        sub = getattr(node, attr, None)
        if sub is not None and not isinstance(sub, str):
            out.append(sub)
    return out


def term_formulas_here(t: Any) -> tuple:
    if isinstance(t, At):
        return (t.when,)
    if isinstance(t, CountSince):
        return (t.count, t.since)
    return ()


def past_nodes(formula: Any) -> tuple:
    """The past-time nodes of ``formula``, innermost first, each once."""
    seen: dict = {}

    def walk(node: Any, inside_past: bool) -> None:
        if isinstance(node, (bool, str)) or node is None:
            return
        if isinstance(node, ForEach):
            walk(node.child, inside_past)
            return
        is_past = isinstance(node, PAST_FORMULAS + PAST_TERMS)
        if inside_past and isinstance(node, _FUTURE):
            raise ValueError(f"a future operator inside a past-time one has no value yet: {node!r}")
        for sub in _children(node):
            walk(sub, inside_past or is_past)
        if is_past and node not in seen:
            seen[node] = None

    walk(formula, False)
    return tuple(seen)


def has_past(formula: Any) -> bool:
    return bool(past_nodes(formula))


def _holds(f: Any, state: dict) -> bool:
    from contragent.formulas.dfa_evaluator import _eval_pointwise

    return bool(_eval_pointwise(f, state))


class PastTracker:
    """The values of a formula's past-time nodes, advanced one event at a time.

    ``step(valuation)`` returns a copy of the valuation with
    ``valuation[PAST_KEY]`` holding every node's value at that event.
    """

    def __init__(self, formula: Any) -> None:
        self.nodes = past_nodes(formula)
        self.reset()

    def reset(self) -> None:
        # Per node, what the next event needs: the previous value of an
        # operand, a running flag, a window of values, a frozen value.
        self._mem: dict = {n: self._initial(n) for n in self.nodes}
        self._first = True

    @staticmethod
    def _initial(n: Any) -> Any:
        if isinstance(n, Y):
            return False  # the operand's value at the previous event
        if isinstance(n, H):
            return True
        if isinstance(n, (S, O)):
            return False
        if isinstance(n, Prev):
            return ()  # the last n values of the term, oldest first
        if isinstance(n, At):
            return (False, None)  # (when has held, the frozen value)
        if isinstance(n, CountSince):
            return None  # count so far, None before ``since`` held
        raise TypeError(type(n).__name__)

    def step(self, valuation: dict) -> dict:
        state = dict(valuation)
        values: dict = {}
        state[PAST_KEY] = values
        mem = self._mem
        for n in self.nodes:  # innermost first: operands' past values are set
            m = mem[n]
            if isinstance(n, Y):
                values[n] = m and not self._first
                mem[n] = _holds(n.child, state)
            elif isinstance(n, S):
                v = _holds(n.right, state) or (m and _holds(n.left, state))
                values[n] = mem[n] = v
            elif isinstance(n, O):
                values[n] = mem[n] = m or _holds(n.child, state)
            elif isinstance(n, H):
                values[n] = mem[n] = m and _holds(n.child, state)
            elif isinstance(n, Prev):
                values[n] = m[0] if len(m) == n.n else None
                mem[n] = (m + (n.term.evaluate(state),))[-n.n :]
            elif isinstance(n, At):
                held, frozen = m
                if _holds(n.when, state):
                    held, frozen = True, n.term.evaluate(state)
                values[n] = frozen if held else None
                mem[n] = (held, frozen)
            elif isinstance(n, CountSince):
                if _holds(n.since, state):
                    m = 0
                elif m is not None and _holds(n.count, state):
                    m += 1
                values[n] = mem[n] = m
        self._first = False
        return state

    def snapshot(self) -> tuple:
        return (dict(self._mem), self._first)

    def restore(self, snap: tuple) -> None:
        mem, self._first = snap
        self._mem = dict(mem)

    def idle(self, idle_valuation: dict, limit: int = 64) -> dict:
        """The valuation of the idle events after the end, past values included.

        The idle event repeats forever, so the past-time values settle after
        at most as many events as the deepest window; the settled valuation
        is returned. A count that keeps growing on idle events (a ``count``
        operand that holds when nothing happens) is read after ``limit``
        idle events. Does not change the tracker.
        """
        saved = self.snapshot()
        try:
            prev = self.step(idle_valuation)
            for _ in range(limit):
                cur = self.step(idle_valuation)
                if cur[PAST_KEY] == prev[PAST_KEY]:
                    return cur
                prev = cur
            return prev
        finally:
            self.restore(saved)


# Annotated traces, kept so that evaluating a formula again on the same
# growing trace (the verifier's recursive backend) annotates only the new
# events. Keyed by the trace object and the formula's past-time nodes.
_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 64


def annotate(formula: Any, trace: list[dict]) -> list[dict]:
    """``trace`` with every event carrying the values of ``formula``'s past-time nodes."""
    nodes = past_nodes(formula)
    if not nodes or not trace:
        return trace
    key = (id(trace), nodes)
    hit = _CACHE.get(key)
    if hit is not None:
        raw, seen, out, tracker = hit
        # The same event objects with the same contents: an event changed
        # in place since (a valuation corrected after the fact) starts over.
        if len(raw) <= len(trace) and all(
            a is b and c == b for a, b, c in zip(raw, trace[: len(raw)], seen, strict=True)
        ):
            for ev in trace[len(raw) :]:
                out.append(tracker.step(ev))
                raw.append(ev)
                seen.append(dict(ev))
            _CACHE.move_to_end(key)
            return list(out)
    tracker = PastTracker(formula)
    out = [tracker.step(ev) for ev in trace]
    _CACHE[key] = (list(trace), [dict(ev) for ev in trace], out, tracker)
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return list(out)


def is_annotated(formula: Any, trace: list[dict]) -> bool:
    nodes = past_nodes(formula)
    if not nodes or not trace:
        return True
    memo = trace[0].get(PAST_KEY)
    return isinstance(memo, dict) and all(n in memo for n in nodes)
