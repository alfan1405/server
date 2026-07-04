"""
app.py — Batik Tradjumas Flask Server
Menggunakan TFLite model untuk deteksi motif batik.
Deploy target: Railway.app
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image

# ── TFLite interpreter ─────────────────────────────────────────────────────
# Railway menggunakan tensorflow penuh (tidak ada tflite-runtime untuk Python 3.11+)
# tensorflow sudah include tensorflow.lite, jadi ini selalu berhasil di Railway.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Konfigurasi model ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "static", "models", "batik_model.tflite")

# Label kelas — urutan HARUS sama dengan urutan saat training
CLASS_LABELS = [
    "JawaBarat_GongSibolong",
    "JawaBarat_MegaMendung",
    "JawaTengah_BokorKencono",
    "JawaTengah_Sidomukti",
    "JawaTengah_Sidomulyo",
    "JawaTengah_Srikaton",
    "JawaTengah_Tribusono",
    "JawaTengah_Truntum",
    "Yogyakarta_Kawung",
    "Yogyakarta_Parang",
    "Yogyakarta_SekarJagad",
    "Yogyakarta_Sidoluhur",
    "Yogyakarta_WahyuTumurun",
    "Yogyakarta_Wirasat",
]

# Normalisasi: 'div255' → [0,1]  atau  'mobilenet' → [-1,1]
# Ganti ke 'mobilenet' jika model dilatih dengan MobileNetV2/EfficientNet
NORMALIZE = "div255"

# ── Lazy-load interpreter ──────────────────────────────────────────────────
_interpreter  = None
_input_idx    = None
_output_idx   = None
_input_h      = None
_input_w      = None


def _get_interpreter():
    global _interpreter, _input_idx, _output_idx, _input_h, _input_w
    if _interpreter is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan batik_model.tflite ada di folder static/models/"
        )

    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    _interpreter = interp
    _input_idx   = inp["index"]
    _output_idx  = out["index"]
    _input_h     = int(inp["shape"][1])
    _input_w     = int(inp["shape"][2])

    print(f"[BatikLens] Model dimuat — input: {_input_h}x{_input_w}, "
          f"{len(CLASS_LABELS)} kelas")


def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((_input_w, _input_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)

    if NORMALIZE == "mobilenet":
        arr = arr / 127.5 - 1.0
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)   # [1, H, W, 3]


def _format_label(raw: str) -> dict:
    """Pisahkan 'JawaBarat_GongSibolong' → {region, motif, display}"""
    parts = raw.split("_", 1)
    if len(parts) == 2:
        region, motif = parts
        # Tambahkan spasi sebelum huruf kapital: GongSibolong → Gong Sibolong
        import re
        motif_spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', motif)
        display = f"{motif_spaced} ({region})"
    else:
        region  = ""
        display = raw
    return {"raw": raw, "display": display, "region": region}


# ══════════════════════════════════════════════════════════════════════════
# Routes — Halaman HTML
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/katalog")
def katalog():
    return render_template("katalog.html")

@app.route("/tradjumasnews")
def tradjumasnews():
    return render_template("tradjumasnews.html")

@app.route("/batikcare")
def batikcare():
    return render_template("batikcare.html")

@app.route("/batiklens")
def batiklens():
    return render_template("batiklens.html")


# ══════════════════════════════════════════════════════════════════════════
# API — Deteksi gambar
# POST /predict
# Body JSON: { "image": "data:image/jpeg;base64,..." }
# ══════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    try:
        _get_interpreter()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    # ── Ambil gambar ──────────────────────────────────────────────────────
    image_bytes = None

    ct = request.content_type or ""

    if "multipart/form-data" in ct:
        # Upload file langsung
        f = request.files.get("image")
        if f is None:
            return jsonify({"error": "Field 'image' tidak ditemukan"}), 400
        image_bytes = f.read()

    elif "application/json" in ct:
        # Base64 data URL dari kamera / FileReader
        payload  = request.get_json(force=True) or {}
        data_url = payload.get("image") or payload.get("imageDataUrl", "")
        if not data_url:
            return jsonify({"error": "Field 'image' kosong"}), 400
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(data_url)
        except Exception:
            return jsonify({"error": "Gagal decode base64 image"}), 400
    else:
        return jsonify({"error": f"Content-Type tidak didukung: {ct}"}), 415

    # ── Inferensi ─────────────────────────────────────────────────────────
    try:
        tensor = _preprocess(image_bytes)
        _interpreter.set_tensor(_input_idx, tensor)
        _interpreter.invoke()

        raw_out = _interpreter.get_tensor(_output_idx)
        scores  = raw_out.reshape(-1).tolist()

        top_idx   = int(np.argmax(scores))
        top_score = float(scores[top_idx])
        top_raw   = CLASS_LABELS[top_idx] if top_idx < len(CLASS_LABELS) else f"Kelas {top_idx}"
        top_info  = _format_label(top_raw)

        # Top-3 kandidat
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        top3 = []
        for i in sorted_idx:
            lbl = CLASS_LABELS[i] if i < len(CLASS_LABELS) else f"Kelas {i}"
            info = _format_label(lbl)
            top3.append({
                "label"      : info["display"],
                "label_raw"  : lbl,
                "region"     : info["region"],
                "score"      : round(scores[i], 4),
            })

        return jsonify({
            "success"    : True,
            "label"      : top_info["display"],
            "label_raw"  : top_raw,
            "region"     : top_info["region"],
            "confidence" : round(top_score, 4),
            "top3"       : top3,
        })

    except Exception as e:
        return jsonify({"error": f"Inferensi gagal: {e}"}), 500


# ── Health check (dipakai Railway untuk cek apakah app hidup) ────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": os.path.exists(MODEL_PATH)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
