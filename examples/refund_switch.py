"""Compare two mid-session switch rules: python -m examples.refund_switch."""

import json
from typing import Literal

from contragent import Event, parse_repr
from contragent.kernel import GroundingState, ground_event, make_evaluator

SwitchRule = Literal["keep_pending", "apply_v2"]
# One request, at the first event of this example. X refers to the next tool call.
V1 = "called('request_refund') -> X(called('check_policy'))"


def refund_trace(rule: SwitchRule, proposal: str = "issue_refund") -> list[dict]:
    """Return the example's audit events. v2 has no refund-check requirement."""
    monitor = make_evaluator(parse_repr(V1))
    state = GroundingState()

    def valuation(tool: str, step: int) -> dict:
        event = Event(ts=step, agent="agent", event_type="tool_call", tool=tool, args={})
        return ground_event(event, step, state)

    monitor.step(valuation("request_refund", 0))
    pending = str(monitor.residual)
    trace = [{"event": "tool_call", "tool": "request_refund", "library": "v1", "residual": pending}]
    if rule == "keep_pending":
        active = [monitor]
    elif rule == "apply_v2":
        active = []
    else:
        raise ValueError(f"Unknown switch rule: {rule}")

    # The switch is an audit annotation between calls. Stepping the DFA here
    # would consume the very next event for which the check is owed.
    trace.append(
        {
            "event": "switch",
            "from": "v1",
            "to": "v2",
            "rule": rule,
            "v1_obligation": pending,
            "disposition": "retained" if active else "released",
        }
    )
    candidate = valuation(proposal, 1)
    verdicts = [m.step(candidate) for m in active]
    trace.append(
        {
            "event": "proposal",
            "tool": proposal,
            "library": "v2",
            "retained_v1_verdicts": verdicts,
            "decision": "block" if "⊥" in verdicts else "allow",
        }
    )
    return trace


if __name__ == "__main__":
    print(
        json.dumps(
            {rule: refund_trace(rule) for rule in ("keep_pending", "apply_v2")},
            indent=2,
            ensure_ascii=False,
        )
    )
