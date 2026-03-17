# strava-mcp-python

A **Python** implementation of the [r-huijts/strava-mcp](https://github.com/r-huijts/strava-mcp) TypeScript server.  
Connects Claude to the Strava API via the **Model Context Protocol (MCP)**.

---

## Features

| Category | Tools |
|---|---|
| 🏃 Athlete | `get-athlete`, `get-athlete-stats`, `get-athlete-zones` |
| 📊 Activities | `list-activities`, `get-activity`, `get-activity-streams`, `get-activity-laps`, `get-activity-kudos`, `get-activity-comments`, `get-starred-segments` |
| 🗺️ Segments | `explore-segments`, `get-segment`, `star-segment`, `unstar-segment`, `get-segment-efforts`, `get-segment-leaderboard` |
| 📍 Routes | `list-routes`, `get-route`, `export-route-gpx`, `export-route-tcx` |

**Total: 16 tools** covering all major Strava API v3 functionality.

---

## Requirements

- Python 3.10+
- A free [Strava API application](https://www.strava.com/settings/api)

---

## Installation

```bash
# Clone or copy this folder
cd strava-mcp-python

# Install dependencies
pip install -r requirements.txt

# Copy and edit the env template
cp .env.example .env
```

---

## Authentication (one-time setup)

1. Go to [https://www.strava.com/settings/api](https://www.strava.com/settings/api)
2. Create a new application
3. Set **"Authorization Callback Domain"** to `localhost`
4. Note your **Client ID** and **Client Secret**
5. Run the setup script:

```bash
python scripts/setup_auth.py
```

Your credentials are saved to `~/.config/strava-mcp/config.json` and `.env`.  
**You only need to do this once** — tokens refresh automatically.

---

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "strava": {
      "command": "python",
      "args": ["/absolute/path/to/strava-mcp-python/server.py"],
      "env": {
        "STRAVA_CLIENT_ID": "your_client_id",
        "STRAVA_CLIENT_SECRET": "your_client_secret",
        "STRAVA_ACCESS_TOKEN": "your_access_token",
        "STRAVA_REFRESH_TOKEN": "your_refresh_token",
        "STRAVA_TOKEN_EXPIRES_AT": "1234567890"
      }
    }
  }
}
```

Or if your `.env` file is set up, just point to `server.py` — it loads `.env` automatically.

---

## Claude Code / MCP CLI

```bash
claude mcp add strava python /absolute/path/to/server.py
```

---

## Example Prompts

- *"What did I do for exercise this week?"*
- *"Analyze my last cycling workout in detail"*
- *"Find me some challenging climbing segments near Pune, India"*
- *"What are my heart rate zones?"*
- *"Export my 'Weekend Loop' route as a GPX file"*
- *"Show my power distribution from yesterday's ride"*

---

## Project Structure

```
strava-mcp-python/
├── server.py           # MCP server entry point
├── strava_client.py    # Strava API client + OAuth2 token refresh
├── tools/
│   ├── athlete.py      # Profile, stats, zones
│   ├── activities.py   # Activities + streams (with smart chunking)
│   ├── segments.py     # Explore, star, leaderboard
│   └── routes.py       # List, get, GPX/TCX export
├── scripts/
│   └── setup_auth.py   # One-time OAuth2 setup
├── requirements.txt
└── .env.example
```

---

## Token Auto-Refresh

Tokens expire after ~6 hours. The server automatically refreshes them before every API call and saves new tokens back to `~/.config/strava-mcp/config.json` and `.env`.

---

## License

MIT
