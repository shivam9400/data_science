"""
Route Tools
list-routes, get-route, export-route-gpx, export-route-tcx
"""

import os
from pathlib import Path
from mcp.types import Tool


def get_route_tools() -> list[Tool]:
    return [
        Tool(
            name="list-routes",
            description="List the authenticated athlete's saved routes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "per_page": {"type": "integer", "default": 30},
                    "page": {"type": "integer", "default": 1},
                },
                "required": [],
            },
        ),
        Tool(
            name="get-route",
            description="Get detailed information about a specific saved route.",
            inputSchema={
                "type": "object",
                "properties": {
                    "route_id": {
                        "type": "integer",
                        "description": "The Strava route ID.",
                    }
                },
                "required": ["route_id"],
            },
        ),
        Tool(
            name="export-route-gpx",
            description=(
                "Export a Strava route as a GPX file to the local filesystem. "
                "Requires ROUTE_EXPORT_PATH to be set in the environment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "route_id": {
                        "type": "integer",
                        "description": "The Strava route ID to export.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (without extension).",
                    },
                },
                "required": ["route_id"],
            },
        ),
        Tool(
            name="export-route-tcx",
            description=(
                "Export a Strava route as a TCX file to the local filesystem. "
                "Requires ROUTE_EXPORT_PATH to be set in the environment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "route_id": {
                        "type": "integer",
                        "description": "The Strava route ID to export.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (without extension).",
                    },
                },
                "required": ["route_id"],
            },
        ),
    ]


async def _export_route(client, route_id: int, fmt: str, filename: str = None) -> dict:
    """Export a route in GPX or TCX format."""
    export_dir = os.getenv("ROUTE_EXPORT_PATH")
    if not export_dir:
        return {
            "error": (
                "ROUTE_EXPORT_PATH is not set. "
                "Add it to your .env file pointing to an existing directory."
            )
        }

    export_path = Path(export_dir)
    if not export_path.exists():
        return {"error": f"Export directory does not exist: {export_dir}"}

    # Fetch route name for the default filename
    try:
        route_info = await client.get(f"/routes/{route_id}")
        route_name = route_info.get("name", f"route_{route_id}")
    except Exception:
        route_name = f"route_{route_id}"

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in route_name)
    out_filename = f"{filename or safe_name}.{fmt}"
    out_path = export_path / out_filename

    # The Strava export endpoints return raw file content
    import httpx
    url = f"https://www.strava.com/api/v3/routes/{route_id}/exports/{fmt}"
    headers = {"Authorization": f"Bearer {client.access_token}"}
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.get(url, headers=headers)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

    return {
        "success": True,
        "route_id": route_id,
        "format": fmt.upper(),
        "filename": out_filename,
        "path": str(out_path),
        "size_bytes": len(resp.content),
    }


async def handle_route_tool(client, name: str, args: dict) -> dict:
    if name == "list-routes":
        athlete = await client.get("/athlete")
        athlete_id = athlete["id"]
        params = {
            "per_page": args.get("per_page", 30),
            "page": args.get("page", 1),
        }
        return await client.get(f"/athletes/{athlete_id}/routes", params=params)

    if name == "get-route":
        return await client.get(f"/routes/{args['route_id']}")

    if name == "export-route-gpx":
        return await _export_route(
            client, args["route_id"], "gpx", args.get("filename")
        )

    if name == "export-route-tcx":
        return await _export_route(
            client, args["route_id"], "tcx", args.get("filename")
        )

    return {"error": f"Unknown route tool: {name}"}
