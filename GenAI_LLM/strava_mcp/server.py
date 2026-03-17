"""
Strava MCP Server (Python)
A Model Context Protocol server that connects to the Strava API.
"""

import asyncio
import json
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
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


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return all available Strava tools."""
    return (
        get_athlete_tools()
        + get_activity_tools()
        + get_segment_tools()
        + get_route_tools()
    )


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool calls to the appropriate handler."""
    try:
        # Ensure token is fresh before every call
        await client.ensure_token_valid()

        if name in [t.name for t in get_athlete_tools()]:
            result = await handle_athlete_tool(client, name, arguments)
        elif name in [t.name for t in get_activity_tools()]:
            result = await handle_activity_tool(client, name, arguments)
        elif name in [t.name for t in get_segment_tools()]:
            result = await handle_segment_tool(client, name, arguments)
        elif name in [t.name for t in get_route_tools()]:
            result = await handle_route_tool(client, name, arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Tool '{name}' error: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    logger.info("Starting Strava MCP Server (Python)...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
