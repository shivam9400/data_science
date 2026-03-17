"""
Strava OAuth2 Setup Script
Run this once to authenticate and save your tokens.

Usage:
    python scripts/setup_auth.py

You'll need:
- STRAVA_CLIENT_ID in your .env or entered interactively
- STRAVA_CLIENT_SECRET in your .env or entered interactively
"""

import asyncio
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

REDIRECT_URI = "http://localhost:8888/callback"
AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
CONFIG_PATH = Path.home() / ".config" / "strava-mcp" / "config.json"

auth_code = None  # Will be set by the callback handler


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="font-family:sans-serif;text-align:center;padding:40px">
                <h2>&#x2705; Strava Authorization Successful!</h2>
                <p>You can close this tab and return to the terminal.</p>
                </body></html>
            """)
        else:
            error = params.get("error", ["unknown"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body>Authorization failed: {error}</body></html>".encode())

    def log_message(self, format, *args):
        pass  # Silence request logs


def get_credentials():
    client_id = os.getenv("STRAVA_CLIENT_ID")
    client_secret = os.getenv("STRAVA_CLIENT_SECRET")

    if not client_id:
        client_id = input("Enter your Strava Client ID: ").strip()
    if not client_secret:
        client_secret = input("Enter your Strava Client Secret: ").strip()

    return client_id, client_secret


def build_auth_url(client_id: str) -> str:
    scopes = "read,activity:read_all,profile:read_all,segments:read,segments:write"
    return (
        f"{AUTH_URL}"
        f"?client_id={client_id}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&approval_prompt=force"
        f"&scope={scopes}"
    )


async def exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        return resp.json()


def save_config(client_id: str, client_secret: str, token_data: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "client_id": client_id,
        "client_secret": client_secret,
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at": token_data["expires_at"],
        "athlete": {
            "id": token_data.get("athlete", {}).get("id"),
            "name": f"{token_data.get('athlete', {}).get('firstname', '')} {token_data.get('athlete', {}).get('lastname', '')}".strip(),
        },
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"\n✅ Config saved to: {CONFIG_PATH}")

    # Also write a .env file in current directory for convenience
    env_path = Path(".env")
    existing = env_path.read_text() if env_path.exists() else ""
    updates = {
        "STRAVA_CLIENT_ID": client_id,
        "STRAVA_CLIENT_SECRET": client_secret,
        "STRAVA_ACCESS_TOKEN": token_data["access_token"],
        "STRAVA_REFRESH_TOKEN": token_data["refresh_token"],
        "STRAVA_TOKEN_EXPIRES_AT": str(token_data["expires_at"]),
    }
    lines = [l for l in existing.splitlines() if not any(l.startswith(k) for k in updates)]
    for k, v in updates.items():
        lines.append(f"{k}={v}")
    env_path.write_text("\n".join(lines) + "\n")
    print(f"✅ .env updated")


async def main():
    print("=== Strava MCP Python — OAuth2 Setup ===\n")
    print("You need a free Strava API app. Create one at:")
    print("  https://www.strava.com/settings/api")
    print("  Set 'Authorization Callback Domain' to: localhost\n")

    client_id, client_secret = get_credentials()

    auth_url = build_auth_url(client_id)
    print(f"\nOpening browser for authorization...")
    print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    # Start local callback server
    server = HTTPServer(("localhost", 8888), CallbackHandler)
    server.timeout = 60
    print("Waiting for Strava authorization callback (60s timeout)...")

    global auth_code
    while auth_code is None:
        server.handle_request()

    server.server_close()

    if not auth_code:
        print("\n❌ Authorization failed or timed out.")
        return

    print(f"\nExchanging authorization code for tokens...")
    token_data = await exchange_code(client_id, client_secret, auth_code)

    athlete = token_data.get("athlete", {})
    name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()
    print(f"\n✅ Authenticated as: {name or 'Unknown athlete'}")

    save_config(client_id, client_secret, token_data)

    print("\n🎉 Setup complete! You can now run the MCP server:")
    print("   python server.py")


if __name__ == "__main__":
    asyncio.run(main())
