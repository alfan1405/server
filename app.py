"""
app.py — Batik Tradjumas Railway Backend
Menerima gambar dari Hostinger frontend, deteksi motif batik dengan TFLite.
"""
from __future__ import annotations
import base64, io, os, re
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Izinkan semua origin (aman untuk API publik tanpa auth session)
# Batasi ke domain tertentu jika perlu:
# origins=["https://domainanda.com","https://www.domainanda.com"]
CORS(app, resources={r"/predict": {"origins": "*"}})

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "static", "models", "batik_model.tflite")

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

NORMALIZE = "div255"  # "div255"=[0,1]  "mobilenet"=[-1,1]

_interpreter = _input_idx = _output_idx = _input_h = _input_w = None

def _get_interpreter():
    global _interpreter, _input_idx, _output_idx, _input_h, _input_w
    if _interpreter is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    _interpreter = interp
    _input_idx   = inp["index"]
    _output_idx  = out["index"]
    _input_h     = int(inp["shape"][1])
    _input_w     = int(inp["shape"][2])
    print(f"[BatikLens] Model dimuat: {_input_h}x{_input_w}, {len(CLASS_LABELS)} kelas")

def _preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((_input_w, _input_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0 if NORMALIZE == "mobilenet" else arr / 255.0
    return np.expand_dims(arr, axis=0)

def _format_label(raw):
    parts = raw.split("_", 1)
    if len(parts) == 2:
        region, motif = parts
        display = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', motif) + f" ({region})"
    else:
        region, display = "", raw
    return {"display": display, "region": region}

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    try:
        _get_interpreter()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    image_bytes = None
    ct = request.content_type or ""

    if "multipart/form-data" in ct:
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "Field 'image' tidak ditemukan"}), 400
        image_bytes = f.read()
    else:
        payload  = request.get_json(force=True) or {}
        data_url = payload.get("image") or payload.get("imageDataUrl", "")
        if not data_url:
            return jsonify({"error": "Field 'image' kosong"}), 400
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(data_url)
        except Exception:
            return jsonify({"error": "Gagal decode base64"}), 400

    try:
        tensor = _preprocess(image_bytes)
        _interpreter.set_tensor(_input_idx, tensor)
        _interpreter.invoke()
        scores  = _interpreter.get_tensor(_output_idx).reshape(-1).tolist()
        top_idx = int(np.argmax(scores))
        top_raw = CLASS_LABELS[top_idx] if top_idx < len(CLASS_LABELS) else f"Kelas {top_idx}"
        top_info = _format_label(top_raw)

        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        top3 = []
        for i in sorted_idx:
            raw  = CLASS_LABELS[i] if i < len(CLASS_LABELS) else f"Kelas {i}"
            info = _format_label(raw)
            top3.append({"label": info["display"], "region": info["region"], "score": round(scores[i], 4)})

        return jsonify({
            "success"    : True,
            "label"      : top_info["display"],
            "label_raw"  : top_raw,
            "region"     : top_info["region"],
            "confidence" : round(float(scores[top_idx]), 4),
            "top3"       : top3,
        })
    except Exception as e:
        return jsonify({"error": f"Inferensi gagal: {e}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": _interpreter is not None,
        "model_exists": os.path.exists(MODEL_PATH),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
