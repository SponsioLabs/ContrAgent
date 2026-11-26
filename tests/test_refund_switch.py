import runpy
from pathlib import Path

refund_trace = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/refund_switch.py")
)["refund_trace"]


def test_same_refund_proposal_after_the_library_switch():
    kept = refund_trace("keep_pending")
    released = refund_trace("apply_v2")
    assert kept[0] == released[0]
    assert kept[0]["residual"] == "called('check_policy')"
    for trace in (kept, released):
        assert [row["event"] for row in trace] == ["tool_call", "switch", "proposal"]
        assert trace[1]["from"] == "v1" and trace[1]["to"] == "v2"
        assert trace[1]["v1_obligation"] == trace[0]["residual"]
        assert trace[2]["tool"] == "issue_refund" and trace[2]["library"] == "v2"
    assert kept[1]["disposition"] == "retained"
    assert kept[2]["retained_v1_verdicts"] == ["⊥"]
    assert kept[2]["decision"] == "block"
    assert released[1]["disposition"] == "released"
    assert released[2]["retained_v1_verdicts"] == []
    assert released[2]["decision"] == "allow"


def test_check_policy_satisfies_the_carried_obligation():
    result = refund_trace("keep_pending", proposal="check_policy")[-1]
    assert result["retained_v1_verdicts"] == ["⊤"]
    assert result["decision"] == "allow"
