"""
payload_type.py

Rule-based classifier that routes a decoded QR payload string to the
correct analysis module. No ML needed here - patterns are well-defined.
"""

import re


def classify_payload(payload: str) -> str:
    p = payload.strip()
    if re.match(r"^https?://", p, re.IGNORECASE):
        return "url"
    if p.lower().startswith("upi://pay"):
        return "upi"
    if p.upper().startswith("BEGIN:VCARD"):
        return "vcard"
    if p.upper().startswith("WIFI:"):
        return "wifi"
    if re.match(r"^(smsto:|mailto:|tel:)", p, re.IGNORECASE):
        return "contact_action"
    return "text"
