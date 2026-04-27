# chimeralang-mcp

**Give Claude typed confidence, hallucination detection, and constraint enforcement — as native MCP tools.**

ChimeraLang is a programming language built for AI cognition. This MCP server exposes its runtime as **33 tools** Claude can call during any conversation — no Anthropic permission needed, works today with Claude Desktop and Claude Code.

---

## Install

```bash
pip install chimeralang-mcp
# or
uvx chimeralang-mcp
```

### Development quick checks

```bash
pytest -q
python -m pip check
```

`pytest` is configured via `pyproject.toml` to include project root on `PYTHONPATH`.

For token counting internals, these optional environment variables are supported:

- `CHIMERA_TOKEN_CACHE_MAX_ENTRIES` (default `2048`) — bounds in-memory token count cache size.
- `CHIMERA_TOKEN_FALLBACK_LOG_INTERVAL_S` (default `60`) — throttles repeated fallback warning logs.

---

## Claude Desktop Setup

Add to your config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "chimeralang": {
      "command": "uvx",
      "args": ["chimeralang-mcp"]
    }
  }
}
```

Or with pip-installed version:
```json
{
  "mcpServers": {
    "chimeralang": {
      "command": "python",
      "args": ["-m", "chimeralang_mcp"]
    }
  }
}
```

Restart Claude Desktop — 33 ChimeraLang tools are now available.

---

## Tools

### Core Language

| Tool | What it does |
|---|---|
| `chimera_run` | Execute a `.chimera` program string |
| `chimera_typecheck` | Static type-check a `.chimera` program without executing |
| `chimera_prove` | Execute + generate a Merkle-chain integrity proof |

### Confidence & Safety

| Tool | What it does |
|---|---|
| `chimera_confident` | Assert a value meets >= 0.95 confidence threshold |
| `chimera_explore` | Wrap a value as exploratory (hallucination explicitly permitted) |
| `chimera_gate` | Collapse multiple candidates via consensus (majority / weighted_vote / highest_confidence) |
| `chimera_constrain` | Full constraint middleware on any tool result |
| `chimera_safety_check` | Validate content against a safety policy |
| `chimera_ethical_eval` | Evaluate an action against ethical principles |

### Hallucination Detection

| Tool | What it does |
|---|---|
| `chimera_detect` | Hallucination detection — 5 strategies: range, dictionary, semantic, cross_reference, temporal |

### Reasoning & Cognition

| Tool | What it does |
|---|---|
| `chimera_plan_goals` | Decompose a high-level goal into ordered sub-goals |
| `chimera_causal` | Build and query a causal graph (add_edge / query / paths / info) |
| `chimera_deliberate` | Multi-perspective deliberation with Jaccard similarity and divergence scoring |
| `chimera_quantum_vote` | Multi-agent consensus voting with contradiction detection |
| `chimera_metacognize` | Reflect on reasoning quality — computes ECE, overconfidence rate |
| `chimera_self_model` | Maintain a persistent self-model of agent capabilities |
| `chimera_embodied` | Embodied reasoning simulation |
| `chimera_social` | Social reasoning and perspective modelling |

### Knowledge & Memory

| Tool | What it does |
|---|---|
| `chimera_world_model` | Persistent in-session world model (key→value with confidence) |
| `chimera_knowledge` | In-session knowledge base (add / search / list) |
| `chimera_memory` | In-session memory store (store / recall by importance) |

### Token Budget & Cost

| Tool | What it does |
|---|---|
| `chimera_compress` | Compress text using abbreviation/shorthand strategies |
| `chimera_optimize` | Aggressive text extraction (structural + entity + frequency) |
| `chimera_fracture` | Full pipeline — optimize docs + compress messages + quality gate |
| `chimera_score` | Rank messages by importance for lossy compression decisions |
| `chimera_budget` | Report current token usage against a budget |
| `chimera_cost_estimate` | Deterministic cost estimate for any supported model |
| `chimera_cost_track` | Record before/after compression events to the session tracker |
| `chimera_dashboard` | Session-level cost intelligence summary |

### Meta & Audit

| Tool | What it does |
|---|---|
| `chimera_audit` | Session-level call log and confidence summary |
| `chimera_evolve` | Evolve and adapt reasoning strategies |
| `chimera_meta_learn` | Meta-learning across reasoning episodes |
| `chimera_transfer_learn` | Transfer learning across domains |
| `chimera_fracture` | Full compression pipeline with quality gate |

---

## What problem does this solve?

Claude's tool-use loop has no built-in mechanism for:

- **Confidence gating** — only proceed if confidence >= threshold
- **Typed output contracts** — this result must satisfy constraint X before going downstream
- **Genuine consensus detection** — is multi-path agreement real, or trivially identical?
- **Hallucination signals** — structured detection, not just "does it sound right"
- **Trust propagation** — confidence degrades through chained tool calls; nothing tracks it
- **Causal reasoning** — explicit cause→effect graphs with pathway queries
- **Multi-perspective deliberation** — structured disagreement scoring across viewpoints
- **Cost intelligence** — token tracking and compression throughout long sessions

ChimeraLang fills exactly these gaps as a constraint layer sitting between Claude and its tools.

---

## Example prompts

**Gate a value before a critical action:**
> *"Before you submit that form, use chimera_confident to verify you're >= 0.95 confident the data is correct."*

**Consensus across reasoning paths:**
> *"Generate 3 different answers, then use chimera_quantum_vote to collapse to the most reliable one."*

**Hallucination scan on output:**
> *"After you get that search result, run chimera_detect with semantic strategy to check for absolute-certainty markers."*

**Full constraint pipeline:**
> *"Use chimera_constrain on that tool result with min_confidence 0.85 and detect_strategy semantic."*

**Integrity proof for audit:**
> *"Run this reasoning with chimera_prove so we have a tamper-evident trace."*

**End-to-end reasoning pipeline:**
> *"Work through 'Should AI be used in autonomous medical diagnosis?' using chimera_plan_goals → chimera_causal → chimera_deliberate → chimera_quantum_vote → chimera_safety_check → chimera_ethical_eval → chimera_prove → chimera_audit."*

---

## ChimeraLang Quick Reference

### Variable Declaration

Both `val` and `let` are supported:

```chimera
val answer = Confident("Paris", 0.97)
let hypothesis = Explore("maybe dark matter is...", 0.4)
```

### Probabilistic Types

```chimera
emit Confident("verified fact", 0.97)   // >= 0.95 required
emit Explore("hypothesis", 0.60)        // hallucination explicitly permitted
```

### Assertions

```chimera
assert Confident(0.78) > Confident(0.45)
```

### Gate Declaration

```chimera
gate verify(claim: Text) -> Converge<Text>
  branches: 3
  collapse: weighted_vote
  threshold: 0.80
  return claim
end
```

### Logical Operators

Both keyword and symbolic forms are supported:

```chimera
// keyword form
if a > 0.5 and b > 0.5
  emit Confident("both pass", 0.9)
end

// symbolic form (also valid)
if a > 0.5 && b > 0.5
  emit Confident("both pass", 0.9)
end
```

### Hallucination Detection

```chimera
detect temperature_check
  strategy: "range"
  on: temperature
  valid_range: [-50.0, 60.0]
  action: "flag"
end
```

### If / Else

```chimera
if confidence > 0.80
  emit Confident("high confidence result", 0.9)
else
  emit Explore("low confidence — needs review", 0.5)
end
```

### For Loop

```chimera
for item in items
  emit Explore(item, 0.6)
end
```

### Match

```chimera
match verdict
| "pass" => emit Confident("approved", 0.95)
| "fail" => emit Explore("rejected", 0.70)
| _      => emit Explore("unknown", 0.50)
end
```

---

## Changelog

### 0.2.6
- Fixed `UnboundLocalError` in `chimera_cost_track` caused by `log` variable shadowing the module-level logger in the `chimera_audit` handler
- Added `let` as a keyword alias for `val` in variable declarations
- Added `&&` and `||` as lexer tokens (aliases for `and` / `or`)
- Expanded tool count to 33

### 0.2.5
- Initial AGI component suite: causal reasoning, deliberation engine, quantum vote, safety layer, ethical reasoner
- Knowledge base, world model, session memory
- Cost tracking, budget management, dashboard
- Self-model and metacognition tools

---

## Links

- **ChimeraLang core:** [github.com/fernandogarzaaa/ChimeraLang](https://github.com/fernandogarzaaa/ChimeraLang)
- **OpenChimera:** [github.com/fernandogarzaaa/OpenChimera_v1](https://github.com/fernandogarzaaa/OpenChimera_v1)

---

## License

MIT © [Fernando Garza](https://github.com/fernandogarzaaa)
