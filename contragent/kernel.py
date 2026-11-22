"""The kernel interface: what a runtime that drives its own agent loop builds on.

``ContrAgent`` and ``Supervisor`` watch a loop that someone else runs: they
gate each tool call and judge the trace. A runtime that owns the loop does
more with the same contracts: it shows the model only the tools the
contracts allow at this step, takes a step itself when only one is left,
and checks a result before the model sees it. For that it needs the parts
underneath the supervisor:

- **grounding**: turning one event into the valuation the monitors read
  (:func:`ground_event` with a :class:`GroundingState` kept across
  events; :func:`collect_content_atoms` for the content predicates a set
  of formulas mentions; :func:`pred_key` for the key of one predicate;
  :func:`idle_valuation` for an event at which nothing happens);
- **monitors**: a contract's formula as an automaton that is stepped one
  valuation at a time, snapshotted and restored to try a step, and read
  for its residual
  (:func:`make_evaluator`, :class:`DFAEvaluator`,
  :class:`ParametricEvaluator`; :class:`Progressor`, the progression the
  monitors and the satisfiability search share);
- **queries**: whether a formula has a model (:func:`is_satisfiable`) and
  whether a set of residuals can still all be satisfied
  (:func:`completable`);
- **formula analysis**: which atoms the agent controls
  (:func:`agent_controlled_atoms`) and whether a formula demands that
  something eventually happens (:func:`demands_eventuality`);
- **contract files**: one entry of a ContrAgent YAML library as a formula
  (:func:`resolve_entry`; :class:`ParseError` when its text does not
  parse).

The names here, and how they are called, stay as they are within a major
version. The modules they come from (``contragent.tracer.grounding``,
``contragent.formulas.dfa_evaluator`` and the others) may be reorganized
in a minor release: import from ``contragent.kernel``, from
``contragent.formulas`` (the formula constructors) and from ``contragent``
itself.
"""

from contragent.analysis.completion import completable
from contragent.eval_runner import resolve_entry
from contragent.formulas._pred_key import idle_valuation, pred_key
from contragent.formulas.dfa_evaluator import DFAEvaluator, ParametricEvaluator, make_evaluator
from contragent.formulas.formula import agent_controlled_atoms, demands_eventuality
from contragent.formulas.parser import ParseError
from contragent.formulas.sat import Progressor, is_satisfiable
from contragent.tracer.grounding import GroundingState, collect_content_atoms, ground_event

__all__ = [
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
]
