"""
Strava MCP Server (Python) — HTTP Transport for Render deployment
"""

import asyncio
import contextlib
import json
import logging
import os

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from strava_client import StravaClient
from tools.athlete import get_athlete_tools, handle_athlete_tool
from tools.activities import get_activity_tools, handle_activity_tool
from tools.segments import get_segment_tools, handle_segment_tool
from tools.routes import get_route_tools, handle_route_tool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as JR

class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        PUBLIC_PATHS = {"/health", "/authorize"}
        # Skip auth for health check
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        
        secret = os.getenv("MCP_SECRET")
        token = request.headers.get("X-MCP-Secret")
        
        if not secret or token != secret:
            return JR({"error": "Unauthorized"}, status_code=401)
        
        return await call_next(request)
    
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


def create_app() -> Starlette:
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=False,
        stateless=True,
    )

    async def health(request):
        return JSONResponse({"status": "ok", "server": "strava-mcp"})

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    # ── Lifespan: start/stop the session manager properly ─────────────────────
    @contextlib.asynccontextmanager
    async def lifespan(app):
        logger.info("Starting StreamableHTTPSessionManager...")
        async with session_manager.run():
            yield
        logger.info("StreamableHTTPSessionManager stopped.")

    starlette_app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/health", health),
            Route("/authorize", authorize),
            Mount("/mcp", app=handle_mcp),
        ],
    )
    # authenticate
    starlette_app.add_middleware(TokenAuthMiddleware)
    starlette_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ── CORS fix: allow browser requests from any origin ──────────────────
    # starlette_app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["*"],
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    # )

    return starlette_app
    # return Starlette(
    #     lifespan=lifespan,
    #     routes=[
    #         Route("/health", health),
    #         Mount("/mcp", app=handle_mcp),
    #     ],
    # )
    async def authorize(request):
        qp = request.query_params
        
        # Optional hardening: pin client_id and enforce S256 allowed_client_ids = {os.getenv("STRAVA_CLIENT_ID", "procam")}
        if qp.get("client_id") not in allowed_client_ids:
            return JSONResponse({"error": "invalid_client_id"}, status_code=400)

        if qp.get("code_challenge_method", "S256") != "S256": 
            return JSONResponse({"error": "code_challenge_method_must_be_S256"},status_code=400)
        strava_auth_base = "https://www.strava.com/oauth/authorize"
        forward_keys = {
            "client_id", "response_type", "redirect_uri", "scope",
            "state", "code_challenge", "code_challenge_method", "approval_prompt"
        }
        forwarded = {k:v for k,v in qp.items() if k in forward_keys}
        forwarded.setdefault("response_type", "code")
        forwarded.setdefault("approval_prompt", "auto")

        return RedirectResponse)f"{strava_auth_base}?{urlencode(forwarded)}", status_code=302)

def run_stdio():
    from mcp.server.stdio import stdio_server

    async def main():
        logger.info("Starting Strava MCP stdio Server...")
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(main())

if __name__ == "__main__":
    transport = os.getenv("TRANSPORT", "http").lower()
    logger.info(f"Transport mode: {transport}")

    if transport == "http":
        port = int(os.getenv("PORT", 8000))
        logger.info(f"Starting Strava MCP HTTP Server on port {port}...")
        starlette_app = create_app()
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        run_stdio()
