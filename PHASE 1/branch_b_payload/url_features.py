"""
url_features.py

Extracts lexical + domain-reputation features from a URL for the
phishing/scam classifier (Branch B / URL sub-branch).
"""

import re
from urllib.parse import urlparse
from datetime import datetime, timezone

try:
    import Levenshtein
except ImportError:  # graceful fallback if not installed
    Levenshtein = None

try:
    import whois
except ImportError:
    whois = None

SUSPICIOUS_KEYWORDS = [
    "verify", "login", "secure", "account", "update", "confirm",
    "otp", "bank", "refund", "prize", "urgent", "suspended", "claim",
]

SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly"}

KNOWN_BRAND_DOMAINS = [
    "google.com", "amazon.com", "amazon.in", "flipkart.com", "hdfcbank.com",
    "icicibank.com", "sbi.co.in", "paytm.com", "irctc.co.in", "gmail.com",
]

IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def _is_ip(host: str) -> bool:
    return bool(IP_PATTERN.match(host))


def _min_typosquat_distance(host: str):
    if Levenshtein is None:
        return None
    return min(Levenshtein.distance(host, brand) for brand in KNOWN_BRAND_DOMAINS)


def _domain_age_days(host: str):
    if whois is None:
        return None
    try:
        w = whois.whois(host)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).days
    except Exception:
        return None  # WHOIS lookup can fail/timeout - treat as unknown, not an error


def extract_url_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.split(":")[0].lower()
    path_and_query = (parsed.path or "") + "?" + (parsed.query or "")

    subdomain_count = max(host.count(".") - 1, 0)
    typo_dist = _min_typosquat_distance(host)
    age_days = _domain_age_days(host)

    features = {
        "url_length": len(url),
        "host_length": len(host),
        "subdomain_count": subdomain_count,
        "is_ip_host": int(_is_ip(host)),
        "is_https": int(parsed.scheme == "https"),
        "is_shortener": int(host in SHORTENER_DOMAINS),
        "suspicious_keyword_count": sum(k in path_and_query.lower() for k in SUSPICIOUS_KEYWORDS),
        "min_typosquat_distance": typo_dist if typo_dist is not None else -1,
        "domain_age_days": age_days if age_days is not None else -1,
        "has_at_symbol": int("@" in url),          # userinfo obfuscation trick
        "digit_ratio_in_host": (sum(c.isdigit() for c in host) / max(len(host), 1)),
    }
    return features


def url_reasons(features: dict) -> list:
    reasons = []
    if features["is_ip_host"]:
        reasons.append("URL uses a raw IP address instead of a domain name")
    if features["is_shortener"]:
        reasons.append("URL uses a link-shortening service, hiding the real destination")
    if features["suspicious_keyword_count"] >= 2:
        reasons.append("URL path contains multiple phishing-associated keywords (verify/login/otp/...)")
    if 0 <= features["min_typosquat_distance"] <= 2:
        reasons.append("Domain is very close in spelling to a known brand domain (typosquatting)")
    if 0 <= features["domain_age_days"] < 30:
        reasons.append("Domain was registered very recently (under 30 days)")
    if not features["is_https"]:
        reasons.append("Connection is not HTTPS")
    if features["has_at_symbol"]:
        reasons.append("URL contains '@', a common redirect-obfuscation trick")
    return reasons
