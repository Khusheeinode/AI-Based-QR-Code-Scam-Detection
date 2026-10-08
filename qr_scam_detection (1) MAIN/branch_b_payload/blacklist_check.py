"""
blacklist_check.py

Cross-checks a URL against the PhishTank public feed.
Sign up for a free PhishTank API key: https://www.phishtank.com/api_register.php
"""

import requests

PHISHTANK_CHECK_URL = "https://checkurl.phishtank.com/checkurl/"


def check_phishtank(url: str, api_key: str = None, timeout: int = 5) -> dict:
    """
    Returns {"in_blacklist": bool, "checked": bool}
    checked=False means the lookup failed/timed out - treat as "unknown", not "safe".
    """
    try:
        data = {"url": url, "format": "json"}
        if api_key:
            data["app_key"] = api_key
        resp = requests.post(PHISHTANK_CHECK_URL, data=data, timeout=timeout)
        resp.raise_for_status()
        result = resp.json().get("results", {})
        return {"in_blacklist": bool(result.get("in_database") and result.get("valid")), "checked": True}
    except Exception:
        return {"in_blacklist": False, "checked": False}
