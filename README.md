# chimeralang-mcp

**Give Claude typed confidence, hallucination detection, and constraint enforcement as native MCP tools.**

ChimeraLang is a programming language built for AI cognition. This MCP server exposes its runtime as **43 tools** Claude can call during any conversation. No Anthropic permission needed, works today with Claude Desktop and Claude Code.

---

## Install

```bash
pip install chimeralang-mcp
# or
uvx chimeralang-mcp
```

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

Or with a pip-installed version:

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

Restart Claude Desktop. 43 ChimeraLang tools are now available.

---

## Tools

The core language/runtime tools are executable and deterministic. Several higher-level reasoning, safety, and cognition helpers are lightweight local heuristics intended for planning, triage, and guardrails rather than authoritative verification.

Most stateful tools accept an optional `namespace` and persist data to `~/.chimeralang_mcp` (or `CHIMERA_MCP_DATA_DIR`) so agents can carry memory, world state, traces, and cost history across sessions.

### Core Language

| Tool | What it does |
|---|---|
| `chimera_run` | Execute a `.chimera` program string |
| `chimera_typecheck` | Static type-check a `.chimera` program without executing |
| `chimera_prove` | Execute plus generate a Merkle-chain integrity proof |

### Confidence and Safety

| Tool | What it does |
|---|---|
| `chimera_confident` | Assert a value meets the `>= 0.95` confidence threshold |
| `chimera_explore` | Wrap a value as exploratory and explicitly allow uncertainty |
| `chimera_gate` | Collapse multiple candidates via consensus |
| `chimera_constrain` | Full constraint middleware on any tool result |
| `chimera_detect` | Hallucination detection across range, dictionary, semantic, cross-reference, temporal, and confidence-threshold strategies |
| `chimera_safety_check` | Validate content against a safety policy |
| `chimera_ethical_eval` | Evaluate an action against ethical principles |

### Reasoning and Cognition

| Tool | What it does |
|---|---|
| `chimera_plan_goals` | Decompose a high-level goal into ordered sub-goals |
| `chimera_causal` | Build and query a causal graph |
| `chimera_deliberate` | Heuristic multi-perspective deliberation with Jaccard similarity and divergence scoring |
| `chimera_quantum_vote` | Multi-agent consensus voting with contradiction detection |
| `chimera_metacognize` | Reflect on reasoning quality and compute calibration metrics |
| `chimera_self_model` | Maintain a persistent self-model of agent capabilities |
| `chimera_embodied` | Embodied reasoning simulation |
| `chimera_social` | Social reasoning and perspective modelling |
| `chimera_evolve` | Run fitness-ranked candidate selection across generations |
| `chimera_meta_learn` | Record adaptation events and retrieve meta-learning stats |
| `chimera_transfer_learn` | Map concepts across source and target domains |

### Knowledge and Memory

| Tool | What it does |
|---|---|
| `chimera_world_model` | Persistent namespace-scoped world model (key to value with confidence) |
| `chimera_knowledge` | Persistent namespace-scoped knowledge base (add, search, list) |
| `chimera_memory` | Persistent namespace-scoped memory store (store, recall by importance) |

### Provenance and Verification

| Tool | What it does |
|---|---|
| `chimera_claims` | Extract atomic claims from text or an envelope |
| `chimera_verify` | Verify claims against evidence and split supported, unsupported, and contradicted results |
| `chimera_provenance_merge` | Merge multiple result envelopes into one aggregated provenance object |
| `chimera_policy` | Apply reusable constraint profiles like `strict_factual` and `code_review` |
| `chimera_trace` | Inspect persisted result envelopes and trace history |

### Token Budget, Cost, and Workflow

| Tool | What it does |
|---|---|
| `chimera_compress` | Compress text using abbreviation and shorthand strategies |
| `chimera_optimize` | Aggressive text cleanup and extraction passes for large text or code blobs |
| `chimera_fracture` | Full pipeline: optimize docs plus compress messages plus quality gate |
| `chimera_score` | Rank messages by importance for lossy compression decisions |
| `chimera_budget` | Report current token usage against a budget |
| `chimera_cost_estimate` | Deterministic cost estimate for any supported model |
| `chimera_cost_track` | Record before and after compression events to the tracker |
| `chimera_dashboard` | Namespace-level cost intelligence summary |
| `chimera_csm` | Context Session Manager: optimize prompt, compress history, and propose a token budget |
| `chimera_budget_lock` | Lock and track an approved per-turn output budget |
| `chimera_mode` | Recommend a task-relevant subset of the tool inventory |
| `chimera_batch` | Execute multiple Chimera tools in a single MCP call |
| `chimera_summarize` | LLM-free extractive summarizer for long documents |

### Meta and Audit

| Tool | What it does |
|---|---|
| `chimera_audit` | Session-level call log, confidence summary, and persistent audit stats |

---

## What problem does this solve?

Claude's tool-use loop has no built-in mechanism for:

- **Confidence gating** - only proceed if confidence is above a threshold
- **Typed output contracts** - a result must satisfy a constraint before going downstream
- **Genuine consensus detection** - determine whether multi-path agreement is substantive
- **Hallucination signals** - structured detection rather than pure intuition
- **Trust propagation** - confidence and provenance should survive chained tool calls
- **Evidence-backed verification** - explicit claims checked against supplied evidence
- **Persistent reasoning state** - memory, world state, and traces carried across sessions
- **Cost intelligence** - token tracking and compression throughout long sessions

ChimeraLang provides a practical constraint layer between Claude and its tools. The language runtime is the strongest guarantee surface. The higher-level cognition and safety helpers are best treated as lightweight first-pass checks unless you pair them with stronger external evidence.

---

## Example prompts

**Gate a value before a critical action:**  
*"Before you submit that form, use `chimera_confident` to verify you're at least 0.95 confident the data is correct."*

**Consensus across reasoning paths:**  
*"Generate 3 different answers, then use `chimera_quantum_vote` to collapse to the most reliable one."*

**Hallucination scan on output:**  
*"After you get that search result, run `chimera_detect` with semantic strategy to check for absolute-certainty markers."*

**Full constraint pipeline:**  
*"Use `chimera_constrain` on that tool result with `min_confidence=0.85` and `detect_strategy=semantic`."*

**Evidence-backed fact checking:**  
*"Extract claims with `chimera_claims`, verify them against these sources with `chimera_verify`, then apply `chimera_policy` using `strict_factual`."*

**Trace and provenance inspection:**  
*"Merge the envelopes from those two tool calls with `chimera_provenance_merge`, then inspect the latest trace with `chimera_trace`."*

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
emit Confident("verified fact", 0.97)
emit Explore("hypothesis", 0.60)
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
if a > 0.5 and b > 0.5
  emit Confident("both pass", 0.9)
end

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
  emit Explore("low confidence - needs review", 0.5)
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

### 0.4.0
- Add a unified result envelope model with confidence, provenance, transform history, claims, constraints, warnings, and metadata
- Persist namespace-scoped knowledge, memory, world model, self model, meta-learning history, traces, and cost tracking to disk
- Add `chimera_claims`, `chimera_verify`, `chimera_provenance_merge`, `chimera_policy`, and `chimera_trace`
- Expose evidence-backed verification and reusable policy application directly through MCP tools
- Add regression coverage for namespace persistence, claim extraction, verification, provenance merge, policy enforcement, and trace inspection

### 0.3.2
- Preserve structured JSON values through `chimera_confident`, `chimera_explore`, and `chimera_constrain` instead of stringifying them
- Fix multimodal message token estimation fallback so `chimera_cost_estimate`, `chimera_budget`, and `chimera_csm` do not undercount text-array content
- Make `chimera_fracture` stop dropping history once the budget is satisfied instead of over-pruning to the minimum message floor
- Fix `chimera_mode full` to report the real live tool inventory and align README documentation with the current 38-tool surface

### 0.2.7
- Fix `UnboundLocalError` in `chimera_cost_track` caused by `log` variable shadowing the module-level logger in the `chimera_audit` handler
- Add `let` as a keyword alias for `val` in variable declarations
- Add `&&` and `||` as lexer tokens (aliases for `and` and `or`)
- Expand tool count to 33 in the README

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

MIT (c) [Fernando Garza](https://github.com/fernandogarzaaa)
