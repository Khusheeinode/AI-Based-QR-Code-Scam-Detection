"""
text_analysis.py

Heuristic risk scoring for payload types that aren't URL or UPI:
plain text, vCard, WiFi credential strings.
"""

import re

SCAM_KEYWORDS = [
    "urgent", "verify", "suspended", "claim now", "limited time", "act now",
    "kyc update", "account blocked", "won a prize", "congratulations",
    "click here", "otp", "refund",
]

EMBEDDED_URL_PATTERN = re.compile(r"https?://\S+")
EMBEDDED_PHONE_PATTERN = re.compile(r"\b\d{10}\b")


def analyze_text(payload: str) -> dict:
    text = payload.lower()
    reasons = []
    risk = 0.0

    hits = [kw for kw in SCAM_KEYWORDS if kw in text]
    if hits:
        reasons.append(f"Contains scam-associated phrasing: {', '.join(hits[:3])}")
        risk += 0.15 * len(hits)

    if EMBEDDED_URL_PATTERN.search(payload):
        reasons.append("Plain-text payload contains an embedded link - should be re-routed through URL analysis")
        risk += 0.2

    if EMBEDDED_PHONE_PATTERN.search(payload) and any(kw in text for kw in ["call", "contact", "helpline"]):
        reasons.append("Contains an embedded phone number paired with an urgency call-to-action")
        risk += 0.15

    return {"risk_score": min(risk, 1.0), "reasons": reasons}


def analyze_vcard(payload: str) -> dict:
    reasons = []
    risk = 0.0
    if "URL:" in payload.upper():
        url_match = EMBEDDED_URL_PATTERN.search(payload)
        if url_match:
            reasons.append("vCard contains an embedded URL field - should be re-routed through URL analysis")
            risk += 0.2
    if not any(f in payload.upper() for f in ["N:", "FN:"]):
        reasons.append("vCard is missing standard name fields - malformed/unusual structure")
        risk += 0.15
    return {"risk_score": min(risk, 1.0), "reasons": reasons}


def analyze_wifi(payload: str) -> dict:
    # WIFI:T:WPA;S:SSID;P:password;;
    reasons = []
    risk = 0.0
    if "T:NOPASS" in payload.upper() or "T:;" in payload.upper():
        reasons.append("WiFi network is configured with no password - low direct scam relevance, but flagged for user awareness")
        risk += 0.05
    return {"risk_score": min(risk, 1.0), "reasons": reasons}
