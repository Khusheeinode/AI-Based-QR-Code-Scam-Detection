# AI-Based QR Code Scam Detection — Build Guide

This is the step-by-step build order for the hybrid system (CNN image branch
+ multi-payload-type analysis branch + fusion). Code for every step is in
this project folder — this README tells you what order to run things in and
what to fill in yourself.

## 0. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`pyzbar` needs the system `zbar` library:
- Ubuntu/Debian: `sudo apt-get install libzbar0`
- macOS: `brew install zbar`
- Windows: the `pyzbar` wheel bundles it — usually no extra step needed.

---

## Step 1 — Generate the CNN image dataset

```bash
cd data
python generate_dataset.py --n_genuine 600 --n_tampered 600
```

This creates `data/images/genuine/` and `data/images/tampered/` — synthetic
genuine QR codes and QR codes with a simulated sticker-overlay or structural
tamper, with photo-realism augmentation (blur, compression, rotation,
shadow) applied to both classes equally so the CNN can't just learn
"blurry = tampered."

**Inspect a handful of images manually before moving on** — make sure the
tampered ones actually look plausibly tampered and the genuine ones look
clean. Adjust the augmentation parameters in `generate_dataset.py` if not.

Optional: physically print ~20–30 QR codes, tamper a few with real paper
stickers, and photograph them with your phone. Add these to a held-out
`data/images/real_world_test/` folder — not used in training, only for a
final "does this generalize beyond synthetic data" sanity check in your
report.

---

## Step 2 — Train the CNN (Branch A)

```bash
python branch_a_image/train_cnn.py
```

This does transfer learning on MobileNetV2: trains a new classification
head first (frozen backbone), then fine-tunes the last ~30 layers of the
backbone at a low learning rate. Saves to
`branch_a_image/qr_cnn_model.keras`.

Watch the printed `Class order` line — `class_names[1]` should be
`"tampered"` (alphabetical: genuine=0, tampered=1). Everything downstream
assumes label 1 = tampered.

If accuracy is poor, first check dataset quality (Step 1) before touching
the model — a mini project's biggest failure mode is a dataset that's too
easy or too noisy, not model architecture.

---

## Step 3 — Build the URL dataset (Branch B, URL sub-branch)

You need a CSV at `data/urls.csv` with columns `url,label` (1=malicious,
0=legitimate).

- **Malicious URLs**: download the current PhishTank CSV feed
  (https://www.phishtank.com/developer_info.php — free, no login required
  for the static feed) or query their API.
- **Legitimate URLs**: download the Tranco top-1M list
  (https://tranco-list.eu/) and sample a few thousand, prefixed with
  `https://`.

Combine both into `data/urls.csv`, shuffle, and keep classes reasonably
balanced (e.g. 2,000 malicious + 2,000 legitimate is a fine mini-project
size).

---

## Step 4 — Train the URL classifier

```bash
python branch_b_payload/train_url_classifier.py
```

Trains Logistic Regression, Random Forest, and XGBoost, 5-fold
cross-validates each, prints accuracy/precision/recall/F1 for all three,
and saves the best one to `branch_b_payload/url_model.joblib`. This
comparison table is exactly the "benchmarked against Logistic Regression
as a baseline" result your synopsis already promised — use it directly in
your report.

---

## Step 5 — UPI / text / vCard / WiFi branches

These are already implemented as rule-based modules
(`branch_b_payload/upi_analysis.py`, `text_analysis.py`) — no training
needed. Test them directly:

```python
from branch_b_payload.upi_analysis import analyze_upi
print(analyze_upi("upi://pay?pa=refund.claim@oksbi&pn=Refund&am=499&cu=INR"))
```

For your report: document *why* these are rule-based rather than ML (no
public labeled UPI-scam dataset exists at the scale needed for training —
this is a legitimate, defensible design decision, not a shortcut).

---

## Step 6 — PhishTank live blacklist check

Sign up for a free API key at
https://www.phishtank.com/api_register.php, then pass it through:

```python
from branch_b_payload.payload_pipeline import analyze_payload
result = analyze_payload("http://some-url.com", phishtank_api_key="YOUR_KEY")
```

If you skip the key, `blacklist_check.py` still works (PhishTank allows
unauthenticated checks at a lower rate limit) — fine for a demo, get a key
if you'll run many lookups.

---

## Step 7 — Grad-CAM explainability

Already implemented in `branch_a_image/gradcam.py`. Test it once your CNN
is trained:

```python
import tensorflow as tf
from branch_a_image.gradcam import predict_with_gradcam

model = tf.keras.models.load_model("branch_a_image/qr_cnn_model.keras")
score, heatmap = predict_with_gradcam(model, "data/images/tampered/tampered_0000.png")
```

`heatmap` is an RGB numpy array you can save/display — the warm regions
show what the CNN focused on. Save a few example heatmaps for your report;
this is your strongest "explainability" evidence.

---

## Step 8 — Fusion

Already implemented in `fusion/fusion.py`. Test end-to-end on your **four
quadrants** before writing your results section:

1. genuine image + benign payload → should land Low
2. genuine image + malicious payload → should land High (payload branch drives it)
3. tampered image + benign payload → should land Medium/High (image branch drives it)
4. tampered image + malicious payload → should land High (both agree)

If quadrant 3 doesn't come out at least Medium, revisit the
`IMAGE_WEIGHT`/`CNN_HARD_OVERRIDE_THRESHOLD` constants in `fusion.py` — this
is the case that proves your hybrid design adds value over a payload-only
system.

---

## Step 9 — Run the full app

```bash
streamlit run app.py
```

Upload a QR image (or use the webcam), see the decoded payload, the fused
risk band, the plain-language reasons list, and the Grad-CAM heatmap.

---

## Step 10 — Evaluation for your report

- **Branch A**: accuracy/precision/recall/F1 on your held-out CNN test
  split (`train_cnn.py` prints these per epoch — pull the final-epoch
  validation numbers, or add a separate eval script using
  `model.evaluate()` on a fresh test directory).
- **Branch B**: the comparison table from `train_url_classifier.py`.
- **End-to-end**: the four-quadrant test from Step 8, plus a small
  real-world photographed test set if you built one in Step 1.

---

## Suggested week-by-week pacing (for a typical mini-project timeline)

| Week | Task |
|---|---|
| 1 | Steps 0–1: setup, dataset generation, manual QA of images |
| 2 | Steps 2, 7: CNN training + Grad-CAM |
| 3 | Steps 3–4: URL dataset + classifier training/comparison |
| 4 | Steps 5–6: UPI/text branches, PhishTank integration |
| 5 | Steps 8–9: fusion tuning, Streamlit app |
| 6 | Step 10 + report writing, four-quadrant testing, buffer for bugs |
