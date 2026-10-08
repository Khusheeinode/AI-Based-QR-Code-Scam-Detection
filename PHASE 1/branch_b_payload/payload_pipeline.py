"""
payload_pipeline.py

Branch B entry point: takes a decoded QR payload string, routes it to the
right analysis module, and returns a unified risk_score + reasons.
"""

import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from payload_type import classify_payload          # noqa: E402
from url_features import extract_url_features, url_reasons  # noqa: E402
from upi_analysis import analyze_upi                # noqa: E402
from text_analysis import analyze_text, analyze_vcard, analyze_wifi  # noqa: E402
from blacklist_check import check_phishtank          # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "url_model.joblib")
FEATURE_ORDER_PATH = os.path.join(os.path.dirname(__file__), "url_feature_order.joblib")

_url_model = None
_feature_order = None


def _load_url_model():
    global _url_model, _feature_order
    if _url_model is None and os.path.exists(MODEL_PATH):
        _url_model = joblib.load(MODEL_PATH)
        _feature_order = joblib.load(FEATURE_ORDER_PATH)
    return _url_model, _feature_order


def analyze_url_payload(url: str, phishtank_api_key: str = None) -> dict:
    features = extract_url_features(url)
    reasons = url_reasons(features)

    model, feature_order = _load_url_model()
    if model is not None:
        row = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
        ml_prob = float(model.predict_proba(row)[0][1])
    else:
        ml_prob = min(0.15 * features["suspicious_keyword_count"], 1.0)  # fallback if no trained model yet
        reasons.append("(ML model not yet trained - using fallback heuristic score)")

    bl = check_phishtank(url, api_key=phishtank_api_key)
    if bl["in_blacklist"]:
        reasons.append("URL matched an entry in the PhishTank phishing blacklist")

    final_score = max(ml_prob, 0.95 if bl["in_blacklist"] else 0.0)
    return {"risk_score": final_score, "reasons": reasons, "ml_probability": ml_prob,
            "blacklisted": bl["in_blacklist"]}


def analyze_payload(payload: str, phishtank_api_key: str = None) -> dict:
    payload_type = classify_payload(payload)

    if payload_type == "url":
        result = analyze_url_payload(payload, phishtank_api_key)
    elif payload_type == "upi":
        result = analyze_upi(payload)
    elif payload_type == "vcard":
        result = analyze_vcard(payload)
    elif payload_type == "wifi":
        result = analyze_wifi(payload)
    else:
        result = analyze_text(payload)

    result["payload_type"] = payload_type
    return result
