"""
fusion.py

Combines Branch A (CNN image risk) and Branch B (payload risk) into a
single, explainable Low/Medium/High verdict.
"""

IMAGE_WEIGHT = 0.4
PAYLOAD_WEIGHT = 0.6

HARD_OVERRIDE_BLACKLIST_SCORE = 0.95
CNN_HARD_OVERRIDE_THRESHOLD = 0.9

LOW_UPPER = 0.3
MEDIUM_UPPER = 0.7


def band(score: float) -> str:
    if score < LOW_UPPER:
        return "Low"
    if score < MEDIUM_UPPER:
        return "Medium"
    return "High"


def fuse(image_risk_score: float, payload_result: dict) -> dict:
    """
    image_risk_score: float 0-1 from the CNN (Branch A)
    payload_result: dict returned by payload_pipeline.analyze_payload (Branch B)
    """
    payload_risk_score = payload_result["risk_score"]

    weighted = IMAGE_WEIGHT * image_risk_score + PAYLOAD_WEIGHT * payload_risk_score

    final_score = weighted
    override_triggered = None

    if payload_result.get("blacklisted"):
        final_score = max(final_score, HARD_OVERRIDE_BLACKLIST_SCORE)
        override_triggered = "URL blacklist match"
    elif image_risk_score >= CNN_HARD_OVERRIDE_THRESHOLD:
        final_score = max(final_score, MEDIUM_UPPER + 0.01)
        override_triggered = "High-confidence image tampering signal"

    reasons = list(payload_result.get("reasons", []))
    if image_risk_score >= 0.5:
        reasons.append(f"QR image shows visual/structural signs of tampering (CNN confidence {image_risk_score:.2f})")
    if override_triggered:
        reasons.append(f"Risk level raised by hard override: {override_triggered}")

    return {
        "final_score": round(final_score, 3),
        "risk_band": band(final_score),
        "image_risk_score": round(image_risk_score, 3),
        "payload_risk_score": round(payload_risk_score, 3),
        "payload_type": payload_result.get("payload_type"),
        "reasons": reasons,
    }
