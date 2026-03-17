"""
Strava API Client
Handles OAuth2 token refresh and all HTTP requests to Strava API v3.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("strava-client")

STRAVA_API_BASE = "https://www.strava.com/api/v3"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
CONFIG_PATH = Path.home() / ".config" / "strava-mcp" / "config.json"


class StravaClient:
    def __init__(self):
        self.client_id = os.getenv("STRAVA_CLIENT_ID")
        self.client_secret = os.getenv("STRAVA_CLIENT_SECRET")
        self.access_token = os.getenv("STRAVA_ACCESS_TOKEN")
        self.refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")
        self.token_expires_at = int(os.getenv("STRAVA_TOKEN_EXPIRES_AT", "0"))

        # Try loading from config file if env vars are missing
        if not self.access_token:
            self._load_from_config()

        self._http = httpx.AsyncClient(timeout=30.0)

    def _load_from_config(self):
        """Load credentials from ~/.config/strava-mcp/config.json"""
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                self.client_id = data.get("client_id", self.client_id)
                self.client_secret = data.get("client_secret", self.client_secret)
                self.access_token = data.get("access_token", self.access_token)
                self.refresh_token = data.get("refresh_token", self.refresh_token)
                self.token_expires_at = data.get("expires_at", self.token_expires_at)
                logger.info("Loaded credentials from config file.")
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

    def _save_to_config(self):
        """Save refreshed tokens to config file."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.token_expires_at,
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
        logger.info("Saved refreshed tokens to config.")

    async def ensure_token_valid(self):
        """Refresh the access token if expired."""
        if time.time() < self.token_expires_at - 60:
            return  # Token still valid

        if not self.refresh_token or not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Missing OAuth credentials. Run scripts/setup_auth.py to authenticate."
            )

        logger.info("Access token expired — refreshing...")
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                STRAVA_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expires_at = data["expires_at"]
        self._save_to_config()
        logger.info("Token refreshed successfully.")

    async def get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        """Make an authenticated GET request to the Strava API."""
        url = f"{STRAVA_API_BASE}{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        resp = await self._http.get(url, headers=headers, params=params or {})
        resp.raise_for_status()
        return resp.json()

    async def put(self, endpoint: str, data: Optional[dict] = None) -> Any:
        """Make an authenticated PUT request."""
        url = f"{STRAVA_API_BASE}{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        resp = await self._http.put(url, headers=headers, json=data or {})
        resp.raise_for_status()
        return resp.json()

    async def delete(self, endpoint: str) -> Any:
        """Make an authenticated DELETE request."""
        url = f"{STRAVA_API_BASE}{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        resp = await self._http.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "deleted"}
