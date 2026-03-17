"""
Strava MCP Server (Python) — HTTP Transport for Render deployment
"""

import asyncio
import contextlib
import json
import logging
import os

from mcp.server import Server
from mcp.types import Tool, TextContent

from strava_client import StravaClient
from tools.athlete import get_athlete_tools, handle_athlete_tool
from tools.activities import get_activity_tools, handle_activity_tool
from tools.segments import get_segment_tools, handle_segment_tool
from tools.routes import get_route_tools, handle_route_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("strava-mcp")

app = Server("strava-mcp")
client = StravaClient()

ALL_TOOL_NAMES = {
    "athlete": [t.name for t in get_athlete_tools()],
    "activity": [t.name for t in get_activity_tools()],
    "segment": [t.name for t in get_segment_tools()],
    "route": [t.name for t in get_route_tools()],
}


@app.list_tools()
async def list_tools() -> list[Tool]:
    return (
        get_athlete_tools()
        + get_activity_tools()
        + get_segment_tools()
        + get_route_tools()
    )


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        await client.ensure_token_valid()

        if name in ALL_TOOL_NAMES["athlete"]:
            result = await handle_athlete_tool(client, name, arguments)
        elif name in ALL_TOOL_NAMES["activity"]:
            result = await handle_activity_tool(client, name, arguments)
        elif name in ALL_TOOL_NAMES["segment"]:
            result = await handle_segment_tool(client, name, arguments)
        elif name in ALL_TOOL_NAMES["route"]:
            result = await handle_route_tool(client, name, arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Tool '{name}' error: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


def run_http():
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.responses import JSONResponse

    # Try the new import path first, fall back to older path
    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        logger.info("Using StreamableHTTPSessionManager")

        session_manager = StreamableHTTPSessionManager(
            app=app,
            event_store=None,
            json_response=False,
            stateless=True,
        )

        async def handle_mcp(scope, receive, send):
            await session_manager.handle_request(scope, receive, send)

    except ImportError:
        # Older mcp versions use a different path
        from mcp.server.http import create_http_app
        logger.info("Using create_http_app (older mcp)")

        mcp_asgi = create_http_app(app)

        async def handle_mcp(scope, receive, send):
            await mcp_asgi(scope, receive, send)

    async def health(request):
        return JSONResponse({"status": "ok", "server": "strava-mcp"})

    starlette_app = Starlette(
        routes=[
            Route("/health", health),
            Mount("/mcp", app=handle_mcp),
        ]
    )

    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Strava MCP HTTP Server on port {port}...")
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


def run_stdio():
    from mcp.server.stdio import stdio_server

    async def main():
        logger.info("Starting Strava MCP stdio Server...")
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    # Default to HTTP for Render deployment, stdio for local
    transport = os.getenv("TRANSPORT", "http").lower()
    logger.info(f"Transport mode: {transport}")
    if transport == "http":
        run_http()
    else:
        run_stdio()