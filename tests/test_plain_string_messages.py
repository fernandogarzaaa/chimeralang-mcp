"""Regression: tools taking `messages` must accept plain strings, not crash.

chimera_fracture, chimera_budget, chimera_cost_estimate and chimera_csm used to
raise AttributeError ('str' object has no attribute 'get') when a client passed
messages as a list of strings. They now coerce them like chimera_score does.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from chimeralang_mcp.server import _MESSAGE_ITEM_SCHEMA, call_tool, list_tools

STRINGS = ["hello there, can you help me plan the migration?", "sure, step one is backups"]


def _call(name, args):
    result = asyncio.run(call_tool(name, args))
    text = result.content[0].text
    assert not result.is_error, text
    assert "AttributeError" not in text
    return json.loads(text)


@pytest.mark.parametrize("algorithm", ["quantum", "classic"])
def test_fracture_accepts_plain_strings(algorithm):
    out = _call("chimera_fracture", {"messages": STRINGS, "token_budget": 500, "algorithm": algorithm})
    assert isinstance(out, dict)


def test_fracture_lossy_accepts_plain_strings():
    _call("chimera_fracture", {"messages": STRINGS * 5, "token_budget": 5,
                                "allow_lossy": True, "algorithm": "classic"})


def test_budget_counts_plain_strings_like_dicts():
    as_str = _call("chimera_budget", {"messages": STRINGS})
    as_dict = _call("chimera_budget", {"messages": [{"role": "user", "content": m} for m in STRINGS]})
    assert as_str["used_tokens"] > 0
    assert as_str["used_tokens"] == as_dict["used_tokens"]


def test_cost_estimate_accepts_plain_strings():
    out = _call("chimera_cost_estimate", {"messages": STRINGS})
    assert out["input_tokens"] > 0


@pytest.mark.parametrize("algorithm", ["quantum", "classic"])
def test_csm_accepts_plain_strings(algorithm):
    _call("chimera_csm", {"prompt": "plan the migration", "messages": STRINGS, "algorithm": algorithm})


def test_message_schemas_declare_items():
    tools = {t.name: t for t in asyncio.run(list_tools())}
    for name in ("chimera_fracture", "chimera_budget", "chimera_score",
                 "chimera_cost_estimate", "chimera_csm"):
        prop = tools[name].input_schema["properties"]["messages"]
        assert prop["items"] == _MESSAGE_ITEM_SCHEMA, name
        assert set(prop["items"]["properties"]) == {"role", "content"}
