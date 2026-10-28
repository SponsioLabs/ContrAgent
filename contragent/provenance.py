"""Which library a trace was judged under.

A saved trace records what the supervisor decided under the library it
ran with. Replaying the trace under another library answers a different
question: what that library would flag on the same events. To keep the
two apart, the trace carries a fingerprint of its library (one digest
per contract and one for the whole) and a snapshot of its formulas, so a
replay can say whether the library changed, what changed, and can
rebuild the original library when its file has since been edited.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from contragent.formulas.det import DetFormula


def _formula(x: Any) -> Any:
    return x.formula if isinstance(x, DetFormula) else x


def _canonical(contract: Any) -> list:
    def one(x: Any) -> list:
        if isinstance(x, DetFormula):
            return [repr(x.formula), x.kind, x.liveness, repr(x.args)]
        return [repr(x)]

    return [
        contract.desc,
        [one(a) for a in contract.assumptions],
        [one(g) for g in contract.guarantees],
    ]


def _digest(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _labels(contracts: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, c in enumerate(contracts):
        label = c.desc or f"contract {i}"
        n = seen.get(label, 0)
        seen[label] = n + 1
        out.append(label if n == 0 else f"{label} ({n + 1})")
    return out


def library_fingerprint(contracts: list[Any]) -> dict:
    """``{"digest": ..., "contracts": {label: digest}}`` for ``contracts``.

    Contracts are labelled by their description. The digest covers the
    formulas and how they are classified (safety or liveness), not the
    enforcement policy, which changes what the supervisor does about a
    violation but not whether a trace violates a contract.
    """
    per = {
        label: _digest(_canonical(c))
        for label, c in zip(_labels(contracts), contracts, strict=True)
    }
    return {"digest": _digest(sorted(per.items())), "contracts": per}


def library_snapshot(contracts: list[Any]) -> list[dict]:
    """The contracts' formulas in ``repr`` form, which :func:`parse_repr` reads back."""
    return [
        {
            "desc": c.desc,
            "assumption": [repr(_formula(a)) for a in c.assumptions],
            "guarantee": [repr(_formula(g)) for g in c.guarantees],
        }
        for c in contracts
    ]


def contracts_from_snapshot(snapshot: list[dict]) -> list[dict]:
    """Contract mappings for ``ContrAgent(contracts=...)`` rebuilt from a snapshot."""
    from contragent.formulas.parser import parse_repr

    return [
        {
            "desc": s.get("desc"),
            "assumption": [parse_repr(a) for a in s.get("assumption", [])] or None,
            "guarantee": [parse_repr(g) for g in s["guarantee"]],
        }
        for s in snapshot
    ]


def diff_fingerprints(recorded: dict | None, current: dict) -> dict:
    """Compare a trace's recorded fingerprint with the replaying library's.

    ``same`` is ``None`` when the trace carries no fingerprint (it was
    saved before fingerprints were recorded), so nothing can be said.
    """
    if not recorded:
        return {"same": None, "added": [], "removed": [], "changed": []}
    old, new = recorded.get("contracts", {}), current["contracts"]
    return {
        "same": recorded.get("digest") == current["digest"],
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
    }
