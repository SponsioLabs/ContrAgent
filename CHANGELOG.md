# Changelog

All notable changes to ContrAgent are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Changed
- License changed from BSD-3-Clause to Apache-2.0.
- A refusal now reports the contract's `desc` ("identity must be
  verified before funds move") instead of echoing the formula. A
  formula without a description of its own inherits the contract's, and
  a contract without any is reported under its formula read back in
  words (`contragent.config.describe_formula`).

- The CHASE bridge checks an export with `chase_cps`, CHASE's Python
  package (chase-cps 0.3, whose checks take `context=`; the `chase`
  extra installs it): `ChaseSession` reads the export with
  `chase_cps.parse_spec` and offers `verify`, `refines`,
  `remove_redundant`, `unsat_core`, and `conjoin`, which return the
  results of `chase_cps` with their status, traces, and cores, and read
  a finite-trace export on finite traces by passing its axiom as the
  `context` of each check.
  `ChaseSession(text)` takes the text of the export;
  `ChaseSession.from_file` and `from_library` build one from a file or a
  library. `PychaseTranslator` and the console methods `run` and
  `refinement` are gone with the raw `pychase` calls.
- A finished session is decided the same way by the runtime monitor
  (`DFAEvaluator.finalize`), the offline replay (the new
  `contragent.formulas.evaluator.evaluate_final`), the satisfiability
  and conflict checks, and the CHASE export: what a contract still owes
  at the end (the next event of an `X`, the event an `F` or a `U` waits
  for) is read on idle events after the last one, at which no tool is
  called and the state stays what it was; a `G` binds the events of the
  session. An obligation to call a tool then fails, a prohibition holds,
  and a fact reads as it was last. Before, the three readings differed
  at the end: replay held a pending `X(called(t))`, the monitor failed
  it but held `X(!called(t))`, the search failed both, and a pending
  `X(X(called(t)))` held, so stopping early passed. `finish_session`
  decides every guarantee still owed, an `X` included, not only the
  liveness ones.

### Fixed
- The CHASE export declares every integer with a range. The logics
  language gives an integer without one the values 0..16, and 710
  integers of the SOP-Bench libraries had none, so a comparison with a
  larger constant (`university.yaml` compares with 90) was checked on
  the wrong values. A counter ranges over `(0:K)` and any other integer
  over its compared constants widened by one. Integers compared with each
  other share one range, the union of their constants widened by the
  number of integers in the group, or `(0:n-1)` for `n` integers compared
  with no constant.
- Contracts built with the fluent helper (`contract(...).guarantees(...)`)
  or from raw formulas now flag their unbounded eventualities, so
  `finish_session()` reports an owed `F` obligation for them as it
  already did for YAML contracts.
- `check_dead_ends` now inspects guarantees loaded from YAML. They
  arrive wrapped in `DetFormula`, which the check passed to its formula
  walkers unopened, so it found no obligation and no count bound and
  reported every loaded library as free of dead ends.
- `parse_repr` reads back `ArgValue`, `ArgLength`, and `CtxValue`. They
  were read as atoms and then as counters (`Var('ArgLength', ...)`)
  that are never set, so a contract written back from its `repr`
  compared the wrong values.

### Removed
- Dead code, none of it referenced by the package, the tests, the
  harnesses, or ACORN: the deprecated `contragent.formulas.fol` module,
  the `Tracer` class of `contragent.tracer` (grounding stays), the
  duplicate `contracts/sopbench/compile_tree.py` (the SOPBench harness
  keeps its copy), the unused baseline-diff types of `eval_runner`, and
  a dozen unreferenced functions and methods (`config_to_guard_kwargs`,
  `make_contracts`, `render_last_turn`, `register_callback`, ...).

### Added
- A saved trace records its library by content: `library_fingerprint`
  (a digest per contract and one for the library), `library_snapshot`
  (the formulas, read back with `parse_repr`), and the ContrAgent
  version; each refused call records where it would have entered the
  trace. `ContrAgent.audit_trace(trace)` and `contragent replay` keep
  what the supervisor recorded apart from what the replaying library
  flags: whether the library is the recorded one, the contracts added,
  removed, or changed, and for each call refused at the time, whether
  the replaying library would refuse it too. `contragent replay
  --original` replays under the snapshot in the trace (#2).
- `contragent.formulas.sat.Progressor`: the satisfiability search's
  abstraction, one progression step, and its end-of-trace verdict
  (`accepting(node, true_keys)`, the true keys of the last event), for
  building explicit automata over atoms.
- Prompts. After a call whose contract owes the next event
  (`G(a -> X b)` once `a` happened), `guard_after` returns a prompt to
  call `b` next (`CheckResult.prompts` / `.prompt`), which `wrap`
  appends to a text result. `guard_finish()`, called before the agent
  stops, returns `allowed=False` with a prompt for every guarantee the
  end would violate, naming the call that would discharge it, at most
  `max_prompts` times a session. Both use the format of a refusal
  ("Policy: …. Call `b` next.") and read the contracts as the end of the
  session does; `"flag"` mode does not prompt.
- `ContrAgent(verbose=True)`: the session printed as a timeline as it
  happens (`contragent.console`).
- `ContrAgent(trace_path=...)` / `save_trace(path)`: the session written
  as a trace file `contragent replay` reads back, with every refusal and
  withheld result recorded in its metadata.
- `ContrAgent.wrap`: a decorator or a tool-table wrapper that puts
  `guard_before` and `guard_after` around every call; a refused call
  returns the refusal text or raises `ContractViolation`.
- `docs/authoring.md`: how to turn a policy into a library (the four
  rule shapes, weak versus strong until, which side a predicate belongs
  on, arguments, and the checks to run before an agent runs under it).
- `tests/test_dead_ends.py` for the static dead-end check whose
  verdicts on the shipped libraries the results ledger reports.
- `examples/`: the README quickstart as a script, a supervised
  tool-calling loop, and two recorded traces for `contragent replay`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this changelog; CI runs the
  examples.
- `contragent.analysis.completion` and `contragent.analysis.dead_ends`:
  the completion verdict and the static dead-end check under the names
  ACORN uses.

## [1.0.0] - 2026-09-21

Reference implementation of *Symbolic Temporal Supervision of LLM Agents
Using Contracts* (arXiv:2609.18128): ALTLf contracts over interaction
predicates, DFA monitors for online enforcement (block, redirect,
escalate, suppress) and offline replay, the conflict check with minimal
unsatisfiable cores, contract formulation from natural language, and the
SOPBench, AgentDojo, R-Judge, and tau²-bench harnesses.

[Unreleased]: https://github.com/yfxiao16/ContrAgent/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yfxiao16/ContrAgent/releases/tag/v1.0.0
