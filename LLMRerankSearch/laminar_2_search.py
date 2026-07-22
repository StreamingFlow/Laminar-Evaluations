"""Self-contained literal search client for the Laminar registry."""

import configparser
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


_AUTH_ID: Optional[str] = None


def _server_url() -> str:
    configured_url = os.getenv("LAMINAR_SERVER_URL")
    if not configured_url:
        config = configparser.ConfigParser()
        config.read(Path(__file__).with_name("config.ini"))
        configured_url = config.get(
            "CONFIGURATION", "SERVER_URL", fallback="http://127.0.0.1:8080"
        )
    return configured_url.rstrip("/")


def _request_json(request: Request) -> Any:
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Laminar request failed with HTTP {error.code}: {detail or error.reason}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Unable to connect to the Laminar server: {error.reason}"
        ) from error

    if not body:
        return []
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError("Laminar returned an invalid JSON response") from error


def _auth_id() -> str:
    global _AUTH_ID
    if _AUTH_ID is not None:
        return _AUTH_ID

    username = os.getenv("LAMINAR_USERNAME")
    password = os.getenv("LAMINAR_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set LAMINAR_USERNAME and LAMINAR_PASSWORD before searching the registry"
        )

    payload = json.dumps({"userName": username, "password": password}).encode("utf-8")
    request = Request(
        f"{_server_url()}/auth/login",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    response = _request_json(request)
    if not isinstance(response, dict):
        raise RuntimeError("Laminar login returned an unexpected response")
    if "ApiError" in response:
        api_error = response["ApiError"]
        message = (
            api_error.get("message", str(api_error))
            if isinstance(api_error, dict)
            else str(api_error)
        )
        raise RuntimeError(f"Laminar login failed: {message}")

    auth_id = response.get("userName")
    if not isinstance(auth_id, str) or not auth_id:
        raise RuntimeError("Laminar login response did not contain a userName")
    _AUTH_ID = auth_id
    return auth_id


def search(query: str, *, client=None) -> List[Dict[str, Any]]:
    """Search workflow and PE names/descriptions and return registry records.

    Reuse an authenticated Laminar client when supplied so the request exactly
    matches the legacy client's URL construction and response handling.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        return []
    if client is not None:
        return client.searchRegistryLiteral(query, search_type="both") or []

    auth_id = quote(_auth_id(), safe="")
    encoded_query = quote(query, safe="")
    url = f"{_server_url()}/registry/{auth_id}/search/{encoded_query}/type/both"
    response = _request_json(Request(url, headers={"Accept": "application/json"}))
    if not isinstance(response, list):
        raise RuntimeError("Laminar search returned an unexpected response")
    if not all(isinstance(item, dict) for item in response):
        raise RuntimeError("Laminar search results contain an unexpected value")
    return response
