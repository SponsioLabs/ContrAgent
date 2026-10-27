"""UnifiedExtractor on Amazon Bedrock: same Messages call as the anthropic
provider, with Bedrock's own body version."""

from __future__ import annotations

import json
from types import SimpleNamespace

from contragent.generation.llm_extraction import UnifiedExtractor


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = {"constraints": [], "tools": [{"name": "issue_refund"}]}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def test_bedrock_sends_the_classic_body_version_and_parses_the_reply():
    client = SimpleNamespace(messages=FakeMessages())
    ex = UnifiedExtractor(provider="bedrock", client=client)
    constraints, tools = ex._call_llm("system", "user")
    assert constraints == [] and tools == [{"name": "issue_refund"}]
    (call,) = client.messages.calls
    assert call["model"] == "us.anthropic.claude-opus-4-6-v1"
    assert call["extra_body"] == {"anthropic_version": "bedrock-2023-05-31", "temperature": 0.0}
    assert "temperature" not in call


def test_the_anthropic_provider_sends_no_sampling_parameters():
    # 2026-10-21 rejects temperature/top_p/top_k on every model
    client = SimpleNamespace(messages=FakeMessages())
    UnifiedExtractor(provider="anthropic", client=client)._call_llm("system", "user")
    call = client.messages.calls[0]
    assert "extra_body" not in call and "temperature" not in call


def test_bedrock_is_never_auto_detected(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert UnifiedExtractor()._provider != "bedrock"


def test_a_reply_with_prose_before_the_fenced_json_is_parsed():
    from contragent.generation.llm_extraction import _json_payload

    reply = 'I will extract the constraints.\n\n```json\n{"constraints": [], "tools": []}\n```'
    assert json.loads(_json_payload(reply)) == {"constraints": [], "tools": []}
    assert json.loads(_json_payload('{"a": 1}')) == {"a": 1}
    assert json.loads(_json_payload('Here: {"a": {"b": 2}} done')) == {"a": {"b": 2}}
