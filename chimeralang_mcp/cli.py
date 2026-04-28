from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Sequence

from chimeralang_mcp.materials.cli import run_materials_cli
from chimeralang_mcp.server import main as run_stdio_server


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

    parsed = parser.parse_args(args)
    if parsed.command == "server":
        if parsed.transport == "http":
            asyncio.run(_run_http_server(parsed.host, parsed.port))
        else:
            run_stdio_server()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
