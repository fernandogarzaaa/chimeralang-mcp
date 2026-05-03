from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Sequence

from chimeralang_mcp.materials.cli import run_materials_cli
from chimeralang_mcp.server import main as run_stdio_server

HOOK_PROMPT_THRESHOLD = 800
HOOK_TOOL_RESPONSE_THRESHOLD = 2000
HOOK_TOOL_INPUT_THRESHOLD = 4000
HOOK_NAMESPACE = "claude-code-hook"
HOOK_INPUT_HEAVY_TOOLS = frozenset({
    "Edit", "MultiEdit", "Write", "NotebookEdit", "Bash",
})
SESSION_POLICY = (
    "Token-saving policy active for this session. Before processing any document, "
    "tool return, or conversation history >500 chars, route it through "
    "chimera_fracture (preferred) or chimera_optimize. Use chimera_budget to monitor "
    "context window usage and chimera_cost_track to log savings. Quantum compression "
    "targets 2-5% of original size while preserving entities and structure."
)


def _get_session_namespace(event: dict[str, Any]) -> str:
    """Derive session-scoped namespace from event payload."""
    session_id = str(event.get("session_id") or event.get("sessionId") or "default")
    return f"{HOOK_NAMESPACE}:{session_id}"


def _hook_emit(event: str, additional_context: str | None) -> None:
    if not additional_context:
        return
    payload = {
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": additional_context,
        }
    }
    json.dump(payload, sys.stdout)


def _read_hook_event() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"prompt": raw}


def _call_optimize_sync(text: str, namespace: str) -> dict[str, Any]:
    from chimeralang_mcp import server as srv

    async def _run() -> dict[str, Any]:
        result = await srv.call_tool(
            "chimera_optimize",
            {"text": text, "namespace": namespace, "level": "medium"},
        )
        for item in result.content:
            payload = getattr(item, "text", None)
            if payload:
                return json.loads(payload)
        return {}

    return asyncio.run(_run())


def _hook_user_prompt() -> int:
    event = _read_hook_event()
    session_namespace = _get_session_namespace(event)
    prompt = str(event.get("prompt", "") or "")
    if len(prompt) < HOOK_PROMPT_THRESHOLD:
        return 0
    try:
        payload = _call_optimize_sync(prompt, session_namespace)
    except Exception as exc:  # pragma: no cover - hook must never break the session
        sys.stderr.write(f"chimeralang hook: optimize failed: {exc}\n")
        return 0
    compressed = payload.get("optimised_text") or ""
    saved = payload.get("estimated_tokens_saved") or 0
    if not compressed or saved <= 0:
        return 0
    summary = (
        "[chimera-token-saver] Prompt was {orig} chars / ~{orig_tok} tokens. "
        "Compressed (ratio={ratio}, saved ~{saved} tokens) — refer to this "
        "condensed form when quoting the prompt back:\n{body}"
    ).format(
        orig=payload.get("original_chars", len(prompt)),
        orig_tok=payload.get("estimated_tokens_before", "?"),
        ratio=payload.get("reduction_ratio", "?"),
        saved=saved,
        body=compressed,
    )
    _hook_emit("UserPromptSubmit", summary)
    return 0


def _hook_session_start() -> int:
    event = _read_hook_event()
    session_namespace = _get_session_namespace(event)
    # Clear session-scoped state to ensure fresh isolation
    try:
        from chimeralang_mcp import server as srv
        srv._dedup_clear(session_namespace)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"chimeralang hook: session_start clear failed: {exc}\n")
    _hook_emit("SessionStart", SESSION_POLICY)
    return 0


def _is_chimera_tool(tool_name: str) -> bool:
    return tool_name.startswith("chimera_") or "chimeralang" in tool_name


def _record_dedup_safely(namespace: str, tool_name: str, tool_input: Any, response_text: str) -> dict | None:
    """Best-effort dedup record. Never raises into the hook."""
    try:
        from chimeralang_mcp import server as srv
        return srv._dedup_record(namespace, tool_name, tool_input, response_text)
    except Exception as exc:  # pragma: no cover - hook must never break the session
        sys.stderr.write(f"chimeralang hook: dedup_record failed: {exc}\n")
        return None


def _lookup_dedup_safely(namespace: str, tool_name: str, tool_input: Any) -> dict | None:
    try:
        from chimeralang_mcp import server as srv
        key = srv._dedup_key(tool_name, tool_input)
        return srv._dedup_lookup(namespace, key)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"chimeralang hook: dedup_lookup failed: {exc}\n")
        return None


def _hook_post_tool_use() -> int:
    event = _read_hook_event()
    session_namespace = _get_session_namespace(event)
    tool_name = str(event.get("tool_name") or "")
    if not tool_name:
        return 0
    response = (
        event.get("tool_response")
        or event.get("response")
        or event.get("output")
        or event.get("result")
    )
    try:
        response_text = response if isinstance(response, str) else json.dumps(response)
    except Exception:
        response_text = str(response or "")
    if _is_chimera_tool(tool_name):
        # Chimera tools self-report budget inline; we still skip dedup-recording
        # them to avoid recursive growth.
        return 0
    tool_input = event.get("tool_input") or event.get("input") or {}
    _record_dedup_safely(session_namespace, tool_name, tool_input, response_text)
    size = len(response_text)
    if size < HOOK_TOOL_RESPONSE_THRESHOLD:
        return 0
    advisory = (
        "[chimera-token-saver] Tool '{tool}' returned ~{size} chars (~{tok} tokens). "
        "Before quoting it back at length, run chimera_optimize on the long excerpts "
        "(preserve_code=true keeps fenced blocks intact). Check chimera_budget if "
        "you're approaching the context window."
    ).format(tool=tool_name, size=size, tok=max(1, size // 4))
    _hook_emit("PostToolUse", advisory)
    return 0


def _input_size(tool_input: Any) -> int:
    if isinstance(tool_input, str):
        return len(tool_input)
    try:
        return len(json.dumps(tool_input, default=str))
    except Exception:
        return len(str(tool_input))


def _hook_pre_tool_use() -> int:
    event = _read_hook_event()
    session_namespace = _get_session_namespace(event)
    tool_name = str(event.get("tool_name") or "")
    if not tool_name or _is_chimera_tool(tool_name):
        return 0
    tool_input = event.get("tool_input") or event.get("input") or {}
    notes: list[str] = []

    # 1. Dedup hit — same tool + same args as a prior call this session.
    prior = _lookup_dedup_safely(session_namespace, tool_name, tool_input)
    if prior and int(prior.get("hit_count") or 0) >= 0 and prior.get("response_chars"):
        notes.append(
            "[chimera-token-saver] '{tool}' was already called with these exact args "
            "(dedup key={key}, response was ~{rc} chars, prior hits={hits}). "
            "Consider chimera_dedup_lookup with action=get,key={key} to confirm "
            "before re-running.".format(
                tool=tool_name,
                key=prior.get("key"),
                rc=prior.get("response_chars"),
                hits=prior.get("hit_count"),
            )
        )

    # 2. Oversized input payload — Edit/Write/Bash with huge bodies.
    if tool_name in HOOK_INPUT_HEAVY_TOOLS:
        size = _input_size(tool_input)
        if size >= HOOK_TOOL_INPUT_THRESHOLD:
            notes.append(
                "[chimera-token-saver] '{tool}' input is ~{size} chars (~{tok} tokens). "
                "If this is mostly natural-language prose or quoted output, consider "
                "running chimera_optimize on the body first (preserve_code=true keeps "
                "code intact). Code-only payloads are usually fine.".format(
                    tool=tool_name, size=size, tok=max(1, size // 4),
                )
            )

    if notes:
        _hook_emit("PreToolUse", "\n\n".join(notes))
    return 0


def _hook_stop() -> int:
    event = _read_hook_event()
    session_namespace = _get_session_namespace(event)
    try:
        from chimeralang_mcp import server as srv

        async def _run() -> dict:
            result = await srv.call_tool(
                "chimera_session_report",
                {"namespace": session_namespace, "include_dedup": True},
            )
            for item in result.content:
                payload = getattr(item, "text", None)
                if payload:
                    return json.loads(payload)
            return {}

        report = asyncio.run(_run())
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"chimeralang hook: session_report failed: {exc}\n")
        return 0
    cost = report.get("cost_summary") or {}
    dedup = report.get("dedup") or {}
    saved = cost.get("total_tokens_saved") or 0
    requests = cost.get("request_count") or 0
    pct = cost.get("avg_pct_saved") or 0.0
    tracked_calls = dedup.get("tracked_calls") or 0
    total_hits = dedup.get("total_hits") or 0
    if saved <= 0 and tracked_calls == 0:
        return 0
    summary = (
        "[chimera-token-saver] Session totals — compression: ~{saved} tokens saved "
        "across {req} request(s), avg {pct}%. Dedup: {calls} unique tool calls "
        "tracked, {hits} repeat hit(s). Use chimera_session_report for details."
    ).format(saved=saved, req=requests, pct=pct, calls=tracked_calls, hits=total_hits)
    _hook_emit("Stop", summary)
    return 0


def _run_hook(event: str) -> int:
    if event == "user-prompt":
        return _hook_user_prompt()
    if event == "session-start":
        return _hook_session_start()
    if event == "post-tool-use":
        return _hook_post_tool_use()
    if event == "pre-tool-use":
        return _hook_pre_tool_use()
    if event == "stop":
        return _hook_stop()
    sys.stderr.write(f"chimeralang hook: unknown event '{event}'\n")
    return 2


async def _run_http_server(host: str, port: int) -> None:
    import anyio
    import uvicorn
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    from chimeralang_mcp.server import server as mcp_server

    transport = StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=True)
    init_options = mcp_server.create_initialization_options()

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["path"] == "/healthz":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"ok": true}',
                }
            )
        elif scope["path"].rstrip("/") == "/mcp":
            await transport.handle_request(scope, receive, send)
        else:
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "not_found"}',
                }
            )

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    web_server = uvicorn.Server(config)

    async with transport.connect() as (read_stream, write_stream):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                mcp_server.run,
                read_stream,
                write_stream,
                init_options,
            )
            tg.start_soon(web_server.serve)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        run_stdio_server()
        return 0

    if args[0] in {"sync", "build", "status", "licenses"}:
        return run_materials_cli(args)

    parser = argparse.ArgumentParser(prog="chimeralang-mcp")
    subparsers = parser.add_subparsers(dest="command")

    server_parser = subparsers.add_parser("server", help="Run the MCP server")
    server_parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8765)

    hook_parser = subparsers.add_parser(
        "hook",
        help="Claude Code hook entry point (reads JSON event from stdin, emits hookSpecificOutput on stdout)",
    )
    hook_parser.add_argument(
        "--event",
        choices=["user-prompt", "session-start", "post-tool-use", "pre-tool-use", "stop"],
        required=True,
    )

    parsed = parser.parse_args(args)
    if parsed.command == "server":
        if parsed.transport == "http":
            asyncio.run(_run_http_server(parsed.host, parsed.port))
        else:
            run_stdio_server()
        return 0
    if parsed.command == "hook":
        return _run_hook(parsed.event)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
