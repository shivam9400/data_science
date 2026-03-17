"""
Activity Tools
list-activities, get-activity, get-activity-streams, get-activity-laps,
get-activity-kudos, get-activity-comments, get-starred-segments
"""

from mcp.types import Tool
import json

MAX_CHUNK_BYTES = 50_000  # ~50KB per chunk for LLM processing


def get_activity_tools() -> list[Tool]:
    return [
        Tool(
            name="list-activities",
            description="List the authenticated athlete's recent activities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "per_page": {
                        "type": "integer",
                        "description": "Number of activities to return (default 30, max 200).",
                        "default": 30,
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (default 1).",
                        "default": 1,
                    },
                    "before": {
                        "type": "integer",
                        "description": "Unix timestamp — return activities before this time.",
                    },
                    "after": {
                        "type": "integer",
                        "description": "Unix timestamp — return activities after this time.",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get-activity",
            description="Get detailed information about a specific Strava activity by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "integer",
                        "description": "The Strava activity ID.",
                    },
                    "include_all_efforts": {
                        "type": "boolean",
                        "description": "Include all segment efforts (default false).",
                        "default": False,
                    },
                },
                "required": ["activity_id"],
            },
        ),
        Tool(
            name="get-activity-streams",
            description=(
                "Fetch detailed time-series data streams for an activity "
                "(power, heart rate, cadence, speed, altitude, etc.). "
                "Returns compact format by default to reduce payload size ~70-80%."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "integer",
                        "description": "The Strava activity ID.",
                    },
                    "stream_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Stream types to request. Options: time, distance, latlng, "
                            "altitude, velocity_smooth, heartrate, cadence, watts, "
                            "temp, moving, grade_smooth. Defaults to all."
                        ),
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "Return compact format (default true). Set false for verbose human-readable format.",
                        "default": True,
                    },
                },
                "required": ["activity_id"],
            },
        ),
        Tool(
            name="get-activity-laps",
            description="Get lap data for a specific activity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "integer",
                        "description": "The Strava activity ID.",
                    }
                },
                "required": ["activity_id"],
            },
        ),
        Tool(
            name="get-activity-kudos",
            description="Get the list of athletes who kudosed a specific activity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "integer",
                        "description": "The Strava activity ID.",
                    },
                    "per_page": {"type": "integer", "default": 30},
                    "page": {"type": "integer", "default": 1},
                },
                "required": ["activity_id"],
            },
        ),
        Tool(
            name="get-activity-comments",
            description="Get the comments on a specific activity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "integer",
                        "description": "The Strava activity ID.",
                    },
                    "per_page": {"type": "integer", "default": 30},
                    "page": {"type": "integer", "default": 1},
                },
                "required": ["activity_id"],
            },
        ),
        Tool(
            name="get-starred-segments",
            description="Get the authenticated athlete's starred segments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "per_page": {"type": "integer", "default": 30},
                    "page": {"type": "integer", "default": 1},
                },
                "required": [],
            },
        ),
    ]


def _compact_streams(raw_streams: list) -> dict:
    """Convert verbose stream objects to compact arrays. Reduces size ~70-80%."""
    compact = {}
    for stream in raw_streams:
        compact[stream["type"]] = {
            "data": stream["data"],
            "series_type": stream.get("series_type"),
            "original_size": stream.get("original_size"),
            "resolution": stream.get("resolution"),
        }
    return compact


def _chunk_if_large(data: dict) -> dict:
    """If payload > MAX_CHUNK_BYTES, add chunking metadata."""
    serialized = json.dumps(data)
    if len(serialized.encode()) <= MAX_CHUNK_BYTES:
        return data

    # Split stream data into chunks
    chunks = []
    keys = list(data.keys())
    chunk_size = max(1, len(keys) // ((len(serialized.encode()) // MAX_CHUNK_BYTES) + 1))

    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i : i + chunk_size]
        chunks.append({k: data[k] for k in chunk_keys})

    return {
        "chunked": True,
        "total_chunks": len(chunks),
        "note": "Large activity split into chunks. Request specific stream types to reduce size.",
        "chunks": chunks,
    }


async def handle_activity_tool(client, name: str, args: dict) -> dict:
    if name == "list-activities":
        params = {k: v for k, v in args.items() if v is not None}
        params.setdefault("per_page", 30)
        params.setdefault("page", 1)
        return await client.get("/athlete/activities", params=params)

    if name == "get-activity":
        activity_id = args["activity_id"]
        params = {"include_all_efforts": args.get("include_all_efforts", False)}
        return await client.get(f"/activities/{activity_id}", params=params)

    if name == "get-activity-streams":
        activity_id = args["activity_id"]
        default_streams = [
            "time", "distance", "latlng", "altitude", "velocity_smooth",
            "heartrate", "cadence", "watts", "temp", "moving", "grade_smooth",
        ]
        stream_types = args.get("stream_types", default_streams)
        compact = args.get("compact", True)

        params = {"keys": ",".join(stream_types), "key_by_type": True}
        raw = await client.get(f"/activities/{activity_id}/streams", params=params)

        if compact:
            # raw is a dict keyed by type when key_by_type=True
            result = {"activity_id": activity_id, "streams": raw, "format": "compact"}
            return _chunk_if_large(result)
        return {"activity_id": activity_id, "streams": raw, "format": "verbose"}

    if name == "get-activity-laps":
        activity_id = args["activity_id"]
        return await client.get(f"/activities/{activity_id}/laps")

    if name == "get-activity-kudos":
        activity_id = args["activity_id"]
        params = {"per_page": args.get("per_page", 30), "page": args.get("page", 1)}
        return await client.get(f"/activities/{activity_id}/kudos", params=params)

    if name == "get-activity-comments":
        activity_id = args["activity_id"]
        params = {"per_page": args.get("per_page", 30), "page": args.get("page", 1)}
        return await client.get(f"/activities/{activity_id}/comments", params=params)

    if name == "get-starred-segments":
        params = {"per_page": args.get("per_page", 30), "page": args.get("page", 1)}
        return await client.get("/segments/starred", params=params)

    return {"error": f"Unknown activity tool: {name}"}
