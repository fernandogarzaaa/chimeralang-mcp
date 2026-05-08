"""ChimeraBench — a corpus of verifiable agent tasks.

Each task is a JSON file in tools/chimerabench/tasks/<family>/<id>.json with
this schema:

    {
      "id":          "unique-slug",
      "family":      "gate-only" | "verify-pipeline" | "deliberate-converge"
                     | "vote-converge" | "mixed-pipeline",
      "description": "human-readable purpose",
      "pipeline": [
        {"tool": "chimera_gate", "args": {...}},
        ...
      ],
      "expected": {
        "program_hashes": ["<sha256 of step 1 envelope>", ...],
        "assertions": [
          {"step": 0, "path": "value",  "equals": "..."},
          {"step": 0, "path": "passed", "equals": true},
          ...
        ]
      },
      "metadata": { ... }
    }

Pipeline semantics: each step is invoked in order; its `provenance.program`
must match the corresponding canonical hash byte-for-byte. After the
pipeline runs, the assertions are evaluated against the per-step results.

This module is intentionally thin — it's a runner, a harness, and a small
self-contained schema. The task corpus lives under tasks/ and is the
substantive artifact.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1"
