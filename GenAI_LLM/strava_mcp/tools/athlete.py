"""
Athlete Tools
get-athlete, get-athlete-stats, get-athlete-zones
"""

from mcp.types import Tool


def get_athlete_tools() -> list[Tool]:
    return [
        Tool(
            name="get-athlete",
            description="Get the authenticated athlete's profile information including name, location, and fitness stats.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get-athlete-stats",
            description="Get the authenticated athlete's activity statistics (totals for runs, rides, swims).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get-athlete-zones",
            description="Get the authenticated athlete's heart rate and power zones.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


async def handle_athlete_tool(client, name: str, args: dict) -> dict:
    if name == "get-athlete":
        return await client.get("/athlete")

    if name == "get-athlete-stats":
        athlete = await client.get("/athlete")
        athlete_id = athlete["id"]
        return await client.get(f"/athletes/{athlete_id}/stats")

    if name == "get-athlete-zones":
        return await client.get("/athlete/zones")

    return {"error": f"Unknown athlete tool: {name}"}
