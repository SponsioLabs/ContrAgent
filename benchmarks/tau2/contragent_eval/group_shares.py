"""Per-rule-group violation shares among tau2-passing traces (paper Figure 7).

Same replay as eval_proc.py. For each (domain, model) cell, reports the
share of tau2-passing simulations that violate at least one contract of
each rule group:

  sameturn  user-facing text and a tool call in one assistant message
  argvalue  an argument value outside the policy's allowed set
  lookup    a required look-up did not precede the action
  limit     a call-count limit

Run with PYTHONPATH=<repo>; traces are expected where eval_proc.py reads them.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_proc as E  # noqa: E402
from convert import tau2_sim_to_trace  # noqa: E402

from contragent.models.trace import Trace  # noqa: E402


def group_of(desc: str) -> str:
    d = desc.lower()
    if "same message as a tool call" in d:
        return "sameturn"
    if "at most" in d or "one-shot" in d:
        return "limit"
    if "must precede" in d:
        return "lookup"
    return "argvalue"


def main():
    honest, _, _ = E.load_classified_contracts()
    files = sorted(p for p in E.RESULTS_DIR.iterdir() if E._FILE_RE.search(p.name))
    order = {"retail": 0, "airline": 1, "telecom": 2}
    out = []
    for path in files:
        m = E._FILE_RE.search(path.name)
        model, domain = m.group("model"), m.group("domain")
        sims = json.loads(path.read_text()).get("simulations", [])
        n_pass = 0
        hits = defaultdict(int)
        for sim in sims:
            td = tau2_sim_to_trace(sim, model=model, domain=domain)
            if not td["metadata"]["tau2_pass"]:
                continue
            n_pass += 1
            fired = E.fire_set_for_trace(Trace.from_dict(td), honest)
            for g in {group_of(c.desc) for c, _ in honest if c.desc in fired}:
                hits[g] += 1
        out.append(
            {
                "domain": domain,
                "model": model,
                "tau2_pass_sims": n_pass,
                "share_pct": {g: 100.0 * hits[g] / n_pass for g in ("sameturn", "argvalue", "lookup", "limit")},
            }
        )
    out.sort(key=lambda r: (order.get(r["domain"], 9), r["model"]))
    print(f"{'domain':8} {'model':28} {'n_pass':>6} {'sameturn':>9} {'argvalue':>9} {'lookup':>7} {'limit':>6}")
    for r in out:
        s = r["share_pct"]
        print(f"{r['domain']:8} {r['model']:28} {r['tau2_pass_sims']:6d} {s['sameturn']:9.1f} {s['argvalue']:9.1f} {s['lookup']:7.1f} {s['limit']:6.1f}")
    dest = HERE / "group_shares.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
