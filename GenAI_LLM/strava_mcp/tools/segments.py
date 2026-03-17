"""
Segment Tools
explore-segments, get-segment, star-segment, unstar-segment,
get-segment-efforts, get-segment-leaderboard
"""

from mcp.types import Tool


def get_segment_tools() -> list[Tool]:
    return [
        Tool(
            name="explore-segments",
            description=(
                "Explore popular cycling or running segments within a geographic bounding box. "
                "Great for finding challenging climbs or popular routes near a location."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bounds": {
                        "type": "string",
                        "description": "Bounding box as 'sw_lat,sw_lng,ne_lat,ne_lng' e.g. '37.7,-122.5,37.8,-122.4'",
                    },
                    "activity_type": {
                        "type": "string",
                        "enum": ["running", "riding"],
                        "description": "Filter by activity type (default: riding).",
                        "default": "riding",
                    },
                    "min_cat": {
                        "type": "integer",
                        "description": "Minimum climb category (0-5).",
                    },
                    "max_cat": {
                        "type": "integer",
                        "description": "Maximum climb category (0-5).",
                    },
                },
                "required": ["bounds"],
            },
        ),
        Tool(
            name="get-segment",
            description="Get detailed information about a specific Strava segment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "integer",
                        "description": "The Strava segment ID.",
                    }
                },
                "required": ["segment_id"],
            },
        ),
        Tool(
            name="star-segment",
            description="Star (save/bookmark) a Strava segment for the authenticated athlete.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "integer",
                        "description": "The segment ID to star.",
                    }
                },
                "required": ["segment_id"],
            },
        ),
        Tool(
            name="unstar-segment",
            description="Unstar a previously starred Strava segment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "integer",
                        "description": "The segment ID to unstar.",
                    }
                },
                "required": ["segment_id"],
            },
        ),
        Tool(
            name="get-segment-efforts",
            description="Get all efforts on a segment by the authenticated athlete.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "integer",
                        "description": "The Strava segment ID.",
                    },
                    "per_page": {"type": "integer", "default": 30},
                    "page": {"type": "integer", "default": 1},
                    "start_date_local": {
                        "type": "string",
                        "description": "ISO 8601 date to filter from (e.g. '2024-01-01T00:00:00Z').",
                    },
                    "end_date_local": {
                        "type": "string",
                        "description": "ISO 8601 date to filter to.",
                    },
                },
                "required": ["segment_id"],
            },
        ),
        Tool(
            name="get-segment-leaderboard",
            description="Get the leaderboard for a Strava segment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "integer",
                        "description": "The Strava segment ID.",
                    },
                    "gender": {
                        "type": "string",
                        "enum": ["M", "F"],
                        "description": "Filter by gender.",
                    },
                    "age_group": {
                        "type": "string",
                        "description": "Age group filter (e.g. '0_24', '25_34', '35_44', '45_54', '55_64', '65_plus').",
                    },
                    "weight_class": {
                        "type": "string",
                        "description": "Weight class filter.",
                    },
                    "following": {
                        "type": "boolean",
                        "description": "Show only athletes the current athlete follows.",
                    },
                    "per_page": {"type": "integer", "default": 10},
                    "page": {"type": "integer", "default": 1},
                },
                "required": ["segment_id"],
            },
        ),
    ]


async def handle_segment_tool(client, name: str, args: dict) -> dict:
    if name == "explore-segments":
        params = {
            "bounds": args["bounds"],
            "activity_type": args.get("activity_type", "riding"),
        }
        if "min_cat" in args:
            params["min_cat"] = args["min_cat"]
        if "max_cat" in args:
            params["max_cat"] = args["max_cat"]
        return await client.get("/segments/explore", params=params)

    if name == "get-segment":
        return await client.get(f"/segments/{args['segment_id']}")

    if name == "star-segment":
        return await client.put(
            f"/segments/{args['segment_id']}/starred", data={"starred": True}
        )

    if name == "unstar-segment":
        return await client.put(
            f"/segments/{args['segment_id']}/starred", data={"starred": False}
        )

    if name == "get-segment-efforts":
        segment_id = args["segment_id"]
        params = {
            "segment_id": segment_id,
            "per_page": args.get("per_page", 30),
            "page": args.get("page", 1),
        }
        if "start_date_local" in args:
            params["start_date_local"] = args["start_date_local"]
        if "end_date_local" in args:
            params["end_date_local"] = args["end_date_local"]
        return await client.get("/segment_efforts", params=params)

    if name == "get-segment-leaderboard":
        segment_id = args["segment_id"]
        params = {
            "per_page": args.get("per_page", 10),
            "page": args.get("page", 1),
        }
        for field in ["gender", "age_group", "weight_class", "following"]:
            if field in args:
                params[field] = args[field]
        return await client.get(f"/segments/{segment_id}/leaderboard", params=params)

    return {"error": f"Unknown segment tool: {name}"}
