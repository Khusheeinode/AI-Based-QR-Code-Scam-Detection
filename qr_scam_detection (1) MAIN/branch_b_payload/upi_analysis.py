"""
upi_analysis.py

Rule-based risk scoring for UPI payment-link payloads
(format: upi://pay?pa=<vpa>&pn=<name>&am=<amount>&cu=<currency>...)

No labeled UPI-scam dataset exists publicly at scale, so this branch is
heuristic/rule-based rather than ML - a legitimate and explainable design
choice for a mini project (document this reasoning in your report).
"""

from urllib.parse import urlparse, parse_qs

KNOWN_PSP_SUFFIXES = {"okhdfcbank", "okaxis", "oksbi", "okicici", "ybl", "paytm", "apl"}

SUSPICIOUS_VPA_KEYWORDS = ["refund", "fine", "penalty", "prize", "reward", "claim", "kyc"]


def parse_upi(payload: str) -> dict:
    parsed = urlparse(payload)
    return {k: v[0] for k, v in parse_qs(parsed.query).items()}


def analyze_upi(payload: str) -> dict:
    fields = parse_upi(payload)
    pa = fields.get("pa", "")
    pn = fields.get("pn", "")
    am = fields.get("am", "")

    reasons = []
    risk = 0.0

    vpa_suffix = pa.split("@")[-1].lower() if "@" in pa else ""
    if vpa_suffix not in KNOWN_PSP_SUFFIXES:
        reasons.append(f"UPI handle suffix '@{vpa_suffix}' is not a recognized bank/PSP handle")
        risk += 0.3

    vpa_local = pa.split("@")[0].lower() if "@" in pa else pa.lower()
    if any(kw in vpa_local for kw in SUSPICIOUS_VPA_KEYWORDS):
        reasons.append("UPI ID contains scam-associated wording (refund/fine/prize/kyc)")
        risk += 0.35

    if vpa_local.replace(" ", "").isdigit() and len(vpa_local) == 10:
        reasons.append("Payee ID is a bare 10-digit mobile number rather than a merchant handle")
        risk += 0.2

    if am:
        reasons.append(f"Amount is pre-filled (₹{am}) - legitimate general-purpose QR usually lets the payer enter the amount")
        risk += 0.15

    if not pn or pn.strip().lower() in {"merchant", "user", "unknown"}:
        reasons.append("Payee name is missing or generic")
        risk += 0.1

    risk = min(risk, 1.0)
    return {"risk_score": risk, "reasons": reasons, "fields": fields}
