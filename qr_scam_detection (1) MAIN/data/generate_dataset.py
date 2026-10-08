"""
generate_dataset.py

Builds the image dataset for Branch A (CNN).
Creates:
  data/images/genuine/*.png      - clean QR codes
  data/images/tampered/*.png     - QR codes with a synthetic "sticker overlay"
                                    or realistic photo-tamper augmentation

Run:
    python generate_dataset.py --n_genuine 600 --n_tampered 600
"""

import os
import random
import argparse
import qrcode
import cv2
import numpy as np
from PIL import Image
import albumentations as A

random.seed(42)
np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
GENUINE_DIR = os.path.join(OUT_DIR, "genuine")
TAMPERED_DIR = os.path.join(OUT_DIR, "tampered")


# ---------- payload generators (reuse for Branch B training data too) ----------

def random_url(benign=True):
    domains_benign = ["amazon.in", "irctc.co.in", "flipkart.com", "hdfcbank.com", "gmail.com"]
    domains_scam = ["paypa1-verify.com", "196.11.23.4/login", "hdfcbannk-secure.tk", "bit.ly/xk93jd"]
    d = random.choice(domains_benign if benign else domains_scam)
    path = random.choice(["", "/login", "/verify-account", "/track", "/pay?amt=500"])
    return f"https://{d}{path}"


def random_upi(benign=True):
    if benign:
        pa = random.choice(["merchant.store@okhdfcbank", "ravi.traders@okaxis", "citycafe@oksbi"])
        am = ""  # legitimate merchant QR usually lets payer type amount
    else:
        pa = random.choice(["refund.claim@oksbi", "parkingfine123@okicici", "9876543210@paytm"])
        am = "&am=499"
    return f"upi://pay?pa={pa}&pn=Merchant{am}&cu=INR"


def random_text(benign=True):
    if benign:
        return random.choice(["Table 12 - Cafe Menu", "Asset ID: LAP-2291", "Wifi guest network info"])
    return random.choice(
        ["URGENT: Your account is suspended, verify now",
         "Claim your prize now, limited time",
         "KYC update required immediately or account blocked"]
    )


def random_payload(benign=True):
    kind = random.choice(["url", "upi", "text"])
    if kind == "url":
        return random_url(benign)
    if kind == "upi":
        return random_upi(benign)
    return random_text(benign)


# ---------- image generation ----------

def make_qr_image(payload: str, size_px: int = 400) -> np.ndarray:
    qr = qrcode.QRCode(border=4, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((size_px, size_px), Image.NEAREST)
    return np.array(img)


def paste_sticker_overlay(base: np.ndarray, overlay_payload: str) -> np.ndarray:
    """Simulate a fake QR sticker pasted over part of a genuine one."""
    h, w, _ = base.shape
    overlay = make_qr_image(overlay_payload, size_px=int(w * random.uniform(0.5, 0.8)))
    oh, ow, _ = overlay.shape

    # slight rotation to mimic a hand-pasted sticker
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((ow // 2, oh // 2), angle, 1.0)
    overlay = cv2.warpAffine(overlay, M, (ow, oh), borderValue=(255, 255, 255))

    x = random.randint(0, w - ow)
    y = random.randint(0, h - oh)
    result = base.copy()
    result[y:y + oh, x:x + ow] = overlay

    # soft shadow at the overlay boundary to simulate a real sticker edge
    shadow = np.zeros_like(result)
    cv2.rectangle(shadow, (x, y), (x + ow, y + oh), (40, 40, 40), thickness=6)
    result = cv2.addWeighted(result, 1.0, shadow, 0.15, 0)
    return result


PHOTO_REALISM = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.GaussianBlur(blur_limit=(1, 3), p=0.4),
    A.ImageCompression(quality_range=(50, 90), p=0.5),
    A.Rotate(limit=6, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.5),
    A.RandomShadow(p=0.3),
])


def generate(n_genuine: int, n_tampered: int):
    os.makedirs(GENUINE_DIR, exist_ok=True)
    os.makedirs(TAMPERED_DIR, exist_ok=True)

    for i in range(n_genuine):
        payload = random_payload(benign=True)
        img = make_qr_image(payload)
        img = PHOTO_REALISM(image=img)["image"]
        Image.fromarray(img).save(os.path.join(GENUINE_DIR, f"genuine_{i:04d}.png"))

    for i in range(n_tampered):
        base_payload = random_payload(benign=True)   # looks genuine at a glance
        overlay_payload = random_payload(benign=False)  # what the scanner actually reads
        base = make_qr_image(base_payload)

        if random.random() < 0.7:
            img = paste_sticker_overlay(base, overlay_payload)
        else:
            # "synthetically malformed" case: irregular module noise, no overlay
            img = base.copy()
            noise_mask = np.random.rand(*img.shape[:2]) < 0.02
            img[noise_mask] = np.random.randint(0, 255, size=(noise_mask.sum(), 3))

        img = PHOTO_REALISM(image=img)["image"]
        Image.fromarray(img).save(os.path.join(TAMPERED_DIR, f"tampered_{i:04d}.png"))

    print(f"Done. {n_genuine} genuine -> {GENUINE_DIR}, {n_tampered} tampered -> {TAMPERED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_genuine", type=int, default=600)
    parser.add_argument("--n_tampered", type=int, default=600)
    args = parser.parse_args()
    generate(args.n_genuine, args.n_tampered)
