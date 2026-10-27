"""``trace_path``: the session is written as a replayable trace with the
decisions in its metadata."""

from __future__ import annotations

import json

from contragent import ContrAgent, Trace


def _session(**kw):
    guard = ContrAgent(agent_id="bank_agent", config="examples/bank.yaml", **kw)
    guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})  # refused
    guard.guard_before("verify_identity", {"user_id": "u1"})
    guard.guard_after("verify_identity", {"ok": True})
    guard.guard_before("transfer_funds", {"amount": 500, "to": "ACME"})
    guard.guard_after("transfer_funds", {"ok": True})
    guard.guard_before("read_file", {"path": "id_rsa"})
    guard.guard_after("read_file", "-----BEGIN PRIVATE KEY-----")  # withheld
    return guard


def test_trace_file_is_written_at_session_end_and_replays(tmp_path):
    path = tmp_path / "session.json"
    guard = _session(trace_path=path)
    assert not path.exists()
    guard.finish_session()
    data = json.loads(path.read_text())
    tools = [e["tool"] for e in data["events"]]
    assert tools == ["verify_identity", "transfer_funds", "read_file"]  # the refused call is absent
    meta = data["metadata"]
    assert meta["agent"] == "bank_agent" and meta["library"] == "bank.yaml"
    assert [d["decision"] for d in meta["decisions"]] == ["refused", "withheld"]
    assert meta["decisions"][0]["contracts"] == ["identity must be verified before funds move"]
    assert meta["pending_obligations"] == ["every transfer is eventually receipted"]

    # The committed events replay clean under the same library: nothing
    # that happened violated a contract, and the receipt is still owed.
    replay = ContrAgent(agent_id="bank_agent", config="examples/bank.yaml")
    report = replay.evaluate_trace(Trace.load(path))
    assert [v["contract"] for v in report["violations"]] == [
        "every transfer is eventually receipted"
    ]


def test_save_trace_on_demand(tmp_path):
    guard = _session()
    out = guard.save_trace(tmp_path / "mid.json")
    data = json.loads(open(out).read())
    assert len(data["events"]) == 3 and data["metadata"]["pending_obligations"] == []


# --- Replaying under another library (issue #2) --------------------------

V2 = """\
version: "1"
agents:
  "*":
    contracts:
      - desc: "no email after a file read"
        G: {ltl: "G((called('read_file') -> G(!(called('send_email')))))"}
      - desc: "at most three transfers per session"
        G: {ltl: "G((Var('count', 'transfer_funds') <= 1))"}
      - desc: "every transfer is eventually receipted"
        G: {ltl: "G((called('transfer_funds') -> F(called('send_receipt'))))"}
      - desc: "no private key reaches the model or leaves in an email"
        A: {ltl: "G(!(output_has('read_file', 'BEGIN PRIVATE KEY')))"}
        G: {ltl: "G(!(arg_field_has('send_email', 'body', 'BEGIN PRIVATE KEY')))"}
      - desc: "files are read only after identity is verified"
        G: {ltl: "(!(called('read_file')) U called('verify_identity')) | G(!(called('read_file')))"}
"""


def _saved(tmp_path):
    path = tmp_path / "session.json"
    _session(trace_path=path).finish_session()
    return path


def test_trace_records_the_library_by_content(tmp_path):
    meta = json.loads(_saved(tmp_path).read_text())["metadata"]
    fp = meta["library_fingerprint"]
    assert len(fp["contracts"]) == 5 and len(fp["digest"]) == 16
    assert meta["contragent"] and len(meta["library_snapshot"]) == 5
    assert meta["decisions"][0]["at"] == 0  # the refused transfer, before any event


def test_audit_under_the_same_library(tmp_path):
    guard = ContrAgent(agent_id="bank_agent", config="examples/bank.yaml")
    audit = guard.audit_trace(Trace.load(_saved(tmp_path)))
    assert audit["library"]["same"] is True
    assert audit["library"]["added"] == audit["library"]["removed"] == []
    assert audit["recorded"]["pending_obligations"] == ["every transfer is eventually receipted"]
    [refusal] = audit["refusals"]
    assert refusal["tool"] == "transfer_funds" and refusal["at"] == 0
    assert refusal["replayed_contracts"] == ["identity must be verified before funds move"]


def test_audit_under_a_changed_library_keeps_the_two_verdicts_apart(tmp_path):
    v2 = tmp_path / "bank_v2.yaml"
    v2.write_text(V2)
    guard = ContrAgent(agent_id="bank_agent", config=v2)
    audit = guard.audit_trace(Trace.load(_saved(tmp_path)))
    lib = audit["library"]
    assert lib["same"] is False
    assert lib["removed"] == ["identity must be verified before funds move"]
    assert lib["added"] == ["files are read only after identity is verified"]
    assert lib["changed"] == ["at most three transfers per session"]
    # What was recorded is unchanged by the replay ...
    assert [d["decision"] for d in audit["recorded"]["decisions"]] == ["refused", "withheld"]
    # ... and the refused transfer, which never ran, v2 would have allowed.
    assert audit["refusals"][0]["replayed_contracts"] == []


def test_snapshot_rebuilds_the_recorded_library(tmp_path):
    from contragent.provenance import contracts_from_snapshot

    meta = json.loads(_saved(tmp_path).read_text())["metadata"]
    rebuilt = ContrAgent(
        agent_id="bank_agent", contracts=contracts_from_snapshot(meta["library_snapshot"])
    )
    assert rebuilt.library_fingerprint() == meta["library_fingerprint"]


def test_snapshot_round_trips_argument_lengths():
    from contragent.provenance import contracts_from_snapshot, library_snapshot

    guard = ContrAgent(config="contragent/contracts/benchmark/tau2_bench.yaml")
    rebuilt = ContrAgent(
        agent_id=guard.agent_id,
        contracts=contracts_from_snapshot(library_snapshot(guard._own_contracts())),
    )
    assert rebuilt.library_fingerprint() == guard.library_fingerprint()


def test_audit_of_a_trace_without_fingerprint(tmp_path):
    path = _saved(tmp_path)
    data = json.loads(path.read_text())
    for key in ("library_fingerprint", "library_snapshot", "contragent"):
        del data["metadata"][key]
    for d in data["metadata"]["decisions"]:
        d.pop("at", None)
    guard = ContrAgent(agent_id="bank_agent", config="examples/bank.yaml")
    audit = guard.audit_trace(Trace.from_dict(data))
    assert audit["library"]["same"] is None
    assert audit["refusals"][0]["replayed_contracts"] is None


def test_replay_cli_reports_the_library(tmp_path):
    from click.testing import CliRunner

    from contragent.cli import cli

    path = _saved(tmp_path)
    v2 = tmp_path / "bank_v2.yaml"
    v2.write_text(V2)
    runner = CliRunner()
    out = runner.invoke(cli, ["replay", str(path), "--config", str(v2)]).output
    assert "DIFFERENT" in out and "removed: identity must be verified" in out
    assert "transfer_funds at event 0: this library would allow it" in out
    out = runner.invoke(cli, ["replay", str(path), "--original"]).output
    assert "snapshot in the trace" in out and "the same contracts" in out
    assert "refused here too: identity must be verified before funds move" in out
    assert runner.invoke(cli, ["replay", str(path)]).exit_code == 2
