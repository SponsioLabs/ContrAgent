"""Argument relations say which way they point.

``arg_check`` was a deny-list and ``scope_check`` an allow-list, so the name
did not tell an extractor the direction; "the note must be exactly X" came
out as "must not be X". The relations are now ``arg_allow`` / ``arg_forbid``
/ ``path_allow`` / ``arg_max_length``, the old names compile as aliases, and
the extraction prompt lists only the new ones.
"""

from __future__ import annotations

from contragent.formulas.formula import Atom, G, Implies, Not
from contragent.generation.structured_ir import (
    RELATION_ALIASES,
    ConstraintIR,
    build_ir_system_prompt,
    compile_ir,
)


def _formula(relation: str, **kw):
    res = compile_ir(ConstraintIR(subject="pay", relation=relation, scope="global", **kw))
    assert not res.error, res.error
    return res.compiled.formula


NOTE = {"field": "note", "patterns": ["^Groceries$"]}


def test_arg_allow_requires_the_value():
    f = _formula("arg_allow", params=NOTE)
    assert f == G(
        Implies(Atom("called", "pay"), Atom("arg_field_has", "pay", "note", "^Groceries$"))
    )


def test_arg_forbid_excludes_the_value():
    f = _formula("arg_forbid", params=NOTE)
    assert f == G(
        Implies(Atom("called", "pay"), Not(Atom("arg_field_has", "pay", "note", "^Groceries$")))
    )


def test_old_names_compile_as_their_new_names():
    cases = {
        "arg_check": {"params": NOTE},
        "scope_check": {"params": {"prefixes": ["/tmp/"]}},
        "length_check": {"params": {"field": "note"}, "quantifier": 40},
    }
    for old, kw in cases.items():
        assert _formula(old, **kw) == _formula(RELATION_ALIASES[old], **kw)


def test_the_prompt_lists_the_new_names_only():
    for mode in ("nl", "document", "code"):
        prompt = build_ir_system_prompt(mode)
        for name in ("arg_allow", "arg_forbid", "path_allow", "arg_max_length"):
            assert f'"{name}"' in prompt, (mode, name)
        for old in RELATION_ALIASES:
            assert old not in prompt, (mode, old)
    assert "must be exactly X" in build_ir_system_prompt("nl")
