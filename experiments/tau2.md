# tau²-bench: procedural violation and outcome scoring (offline)

## Claim

Outcome-based scoring hides procedural violations. With a contract library
built from the benchmark's policy documents, ContrAgent measures procedural
compliance deterministically over every recorded trace and exposes the gap
between passing a task and completing it without violating the procedure.

## Setup

tau²-bench evaluates customer-service agents in three domains (retail,
airline, telecom). Its native metric is `pass^k`: the task reaches the
correct final state in all k retries. AgentPex showed that 83% of
reward-1.0 Claude traces still violate a procedural rule. We replay the
published trace matrix (4,464 traces, 12 model x domain cells) against a
library compiled from each domain's `policy.md` and report `joint^k`: the
outcome is correct and no contract fires, in all k retries.

## Contracts

`contragent/contracts/benchmark/tau2_bench.yaml`: 110 contracts (retail 39,
airline 28, telecom 40, cross-domain 3) covering identity gates, ordering,
argument grounding, forbidden transitions, and the same-turn protocol. Every
contract traces to a sentence of the domain's `policy.md`; ten post-write
re-read contracts that had no policy source were removed in the v2 revision.
Four contracts that compare two argument fields (item-count match, distinct
payment method, passenger cap) are excluded from the reported rates by the
harness, as in the paper. The offline replay evaluates the 43 contracts
(37 distinct rules) that are decidable from tool names, arguments, and
call order alone; contracts that need environment facts (order status,
authentication state) are skipped, see
[`benchmarks/tau2/contragent_eval/README.md`](../benchmarks/tau2/contragent_eval/README.md).

## Results (paper Figures `fig:tau2tax` and `fig:tau2contract`)

Summary file: [`results/tau2_proc_eval.json`](results/tau2_proc_eval.json)
(per-cell `pass^4` / `proc-clean^4` / `joint^4`). Per-rule-group shares
among tau2-passing traces (paper Figure 7):
[`results/tau2_group_shares.json`](results/tau2_group_shares.json).

Across the 4,464-trace matrix Claude 3.7 reaches an outcome `pass^4` of
25% to 60% yet a `joint^4` of 0% in all three domains: no task is completed
reliably without at least one procedural violation. The contract library
decides this deterministically for every trace, at 0.83 ms p50 per call
(appendix latency table), where an LLM judge approximates it.

## Reproduce

Place the tau²-bench trace matrix under
`benchmarks/tau2/data/tau2/results/final/` (the layout of the published
runs), then

```bash
PYTHONPATH=. python benchmarks/tau2/contragent_eval/eval_proc.py
```

`eval_proc.py` writes `proc_eval_results.json` with per-cell fire rates and
the `pass^k` / `proc-clean^k` / `joint^k` matrix;
`group_shares.py` writes the per-rule-group shares behind Figure 7.
