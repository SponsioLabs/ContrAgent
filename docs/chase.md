# Design-time analysis with CHASE

[CHASE](https://chase-cps.github.io) is a contract-based requirement-engineering
framework with a contract algebra (composition, conjunction, refinement) and
model-checking and synthesis back ends. A ContrAgent library exports to CHASE's
*logics* specification language:

```bash
contragent export-chase --config contragent/contracts/sopbench/bank.yaml -o bank.logics
```

The export grounds the library: every instantiated interaction predicate
becomes a proposition, every quantity an integer variable, and each contract a
`CONTRACT` block with its `Assumptions` and `Guarantees`. Every integer is
declared with a range, `integer (lo:hi) variable`, since the logics language
gives an integer without one the values 0..16: a saturating counter ranges over
`(0:K)`, with `K` one more than the largest constant it is compared with, and
any other integer from its smallest to its largest compared constant, widened
by one (an `arg_length` from 0 at the lowest). Integers compared with each other
share the union of their constants, widened by the size of the group, and `n`
integers compared with no constant range over `(0:n-1)`. The ranges keep a value
for every outcome of the comparisons, so a check over them is exact (the module
docstring of `contragent.analysis.chase` gives the argument).

Because CHASE reasons over infinite words, the default `--semantics finite`
writes ContrAgent's reading of a finished session: what a contract still owes at
the end (the next event of an `X`, the event an `F` or a `U` waits for) is read
on idle events after the last one, at which no tool is called and the state stays
what it was, while a `G` binds the events of the session. A proposition `alive`
marks the events of the session, `G` becomes `G(alive -> .)`, and the contract
`finite` asserts that the session is non-empty, that every event proposition is
false once it is over, and that every state proposition and every relation over
state quantities keeps its value from then on. A relation over the arguments of
a call is left free after the end. `--semantics infinite` exports the formulas as
written. Contract blocks are numbered (`c1`, `c2`, ...) and carry their
description in the comment above them, since CHASE's console crashes on contract
names longer than eight characters.

## Checking with chase_cps

With `chase_cps`, CHASE's Python package, and nuXmv installed,
`contragent.analysis.chase.ChaseSession` reads an export with
`chase_cps.parse_spec` and checks its contracts:

```python
from contragent.analysis.chase import ChaseSession

s = ChaseSession.from_file("bank.logics")   # or ChaseSession.from_library(contracts)
s.verify("c1")             # compatibility and consistency: sat, or unsat
s.refines("c2", "c1")      # valid, or invalid with a counterexample trace
s.remove_redundant("c3")   # the contract without its implied terms
core = s.unsat_core()      # a minimal set of terms that cannot hold together
s.contracts_in(core)       # the contracts they come from
```

Each method returns the result of the `chase_cps` check of the same name, with
its status, its counterexample or witness trace, and its core; `conjoin` folds
CHASE conjunction over contracts on one agent. The finite-trace axiom of a
`--semantics finite` export is not a contract of the session: every check reads
the contracts on the traces it allows. `unsat_core` searches the assumptions and
the guarantees of the contracts, the conjunction `contragent conflicts` checks.

The session passes the finite-trace axiom as the `context` of each check, so
it needs chase-cps 0.3, whose checks take `context=`:

```bash
pip install 'chase-cps>=0.3.0,<0.4'    # or pip install 'contragent[chase]'
chase-cps install nuxmv --accept-license
```

## Standalone logics_tool

For a specification read from a logics file, the `verify` command of CHASE's
`logics_tool` emits the NuSMV model with an empty `VAR` block;
`contragent.analysis.chase.smv_declarations` gives the lines that fill it from
the declarations of the export, so the model runs in nuXmv as is. A
`logics_tool` command file terminates each command with a semicolon.
