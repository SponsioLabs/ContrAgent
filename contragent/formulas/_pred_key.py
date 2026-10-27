"""Single source of truth for predicate key formatting.

Both the **formula side** (``Atom.key()``) and the **grounding side**
(``grounding.ground()``) must produce identical string keys for the
evaluator's ``dict.get()`` lookup to work.  This module defines the
canonical format so the two sides cannot drift out of sync.

Format: ``"predicate(arg1, arg2)"`` -- e.g. ``"called(fraud_check)"``
or ``"precedes(fraud_check, execute_refund)"``.

For a predicate whose first argument names a tool (``called``, ``count``,
``arg_field_has``, ...), that argument is written in its canonical
spelling (see :mod:`contragent.formulas.tool_names`). This is the one
place both the formula side and the grounding side pass through, so a
contract written against ``Issue_Refund`` and an event that arrives as
``issue_refund`` meet on the same key.

Example:
    >>> from contragent.formulas._pred_key import pred_key
    >>> pred_key("called", "fraud_check")
    'called(fraud_check)'
    >>> pred_key("precedes", "fraud_check", "execute_refund")
    'precedes(fraud_check, execute_refund)'
"""

# Predicates whose first argument is a tool name.
_TOOL_KEYED = frozenset(
    {
        "called",
        "called_with",
        "count",
        "count_with",
        "consecutive_count",
        "arg_has",
        "arg_field_has",
        "arg_numeric",
        "arg_length_exceeds",
        "arg_paths_within",
        "arg_value",
        "output_has",
    }
)


# Predicates that describe what happens *at* one event: a tool call and
# its arguments and output, an LLM request or response. Grounding emits
# them only on the event they describe. Every other key (counters,
# ``ctx``/``ctx_value`` facts, permissions, ``flow``/``contains`` once
# they occurred, the clock, ``time_since``) describes the state the
# session is in and carries over from event to event.
EVENT_PREDICATES = frozenset(
    {
        "called",
        "called_any",
        "called_with",
        "arg_has",
        "arg_field_has",
        "arg_length_exceeds",
        "arg_numeric",
        "arg_paths_within",
        "arg_value",
        "output_has",
        "llm_said",
        "prompt_contains",
        "system_prompt_present",
        "context_length",
        "segment",
        "response_words",
        "response_chars",
    }
)


def predicate_of(key: str) -> str:
    """The predicate name of a key: ``"called(x)"`` -> ``"called"``, ``"now"`` -> ``"now"``."""
    return key.split("(", 1)[0]


def idle_valuation(valuation: dict) -> dict:
    """The valuation of an event at which nothing happens, after ``valuation``.

    A session that has ended is read as continuing with such events
    forever: no tool is called and nothing is said, so every event
    predicate is absent, and the state stays what it was at the last
    event. This is what grounding emits for an event of no kind.
    """
    return {k: v for k, v in valuation.items() if predicate_of(k) not in EVENT_PREDICATES}


def _escape(s: str) -> str:
    """Escape characters that would be ambiguous in predicate keys."""
    s = str(s)
    return (
        s.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace(",", "\\,")
        .replace(" ", "\\ ")
    )


def pred_key(predicate: str, *args: str) -> str:
    """Build the canonical string key for a predicate.

    Arguments are escaped so that tool names containing ``(``, ``)``,
    or ``,`` do not produce ambiguous keys.

    Args:
        predicate: Predicate name (e.g. ``"called"``, ``"precedes"``).
        *args: Predicate arguments (e.g. tool names).

    Returns:
        A string like ``"called(fraud_check)"`` or
        ``"precedes(fraud_check, execute_refund)"``.
        Zero-argument predicates produce ``"predicate()"``.
    """
    if not args:
        return f"{predicate}()"
    if predicate in _TOOL_KEYED:
        from contragent.formulas.tool_names import canonical_tool

        args = (canonical_tool(args[0]), *args[1:])
    return f"{predicate}({', '.join(_escape(a) for a in args)})"
