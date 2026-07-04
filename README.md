# Batik Tradjumas — Flask + Railway Deploy Guide

---

## Struktur Folder

```
batik-tradjumas/
├── app.py                        ← Flask server (JANGAN UBAH kecuali perlu)
├── requirements.txt              ← dependensi Python
├── Procfile                      ← perintah start untuk Railway
├── railway.toml                  ← konfigurasi Railway
├── .gitignore
│
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── batiklens.js          ← logika frontend (connect ke /predict)
│   │   ├── nav.js
│   │   ├── katalog.js
│   │   └── tradjumasnews.js
│   ├── images/                   ← semua gambar website
│   └── models/
│       └── batik_model.tflite    ← MODEL ANDA TARUH DI SINI
│
└── templates/                    ← file HTML (Jinja2)
    ├── index.html
    ├── katalog.html
    ├── tradjumasnews.html
    ├── batikcare.html
    └── batiklens.html
```

---

## Format Model: Gunakan .tflite

| Format         | Ukuran  | RAM Railway | Cocok Deploy? |
|----------------|---------|-------------|---------------|
| `.tflite`      | Kecil   | ~200–400MB  | ✅ Terbaik    |
| `.keras`       | Sedang  | ~800MB–2GB  | ⚠️ Bisa, mahal |
| `.json + .h5`  | Sedang  | ~800MB–2GB  | ⚠️ Bisa, mahal |

**Gunakan `.tflite`** — lebih ringan, cocok untuk Railway free/hobby tier.

---

## LANGKAH DEPLOY KE RAILWAY

### Prasyarat
- Akun GitHub: https://github.com
- Akun Railway: https://railway.app (daftar gratis pakai GitHub)
- Git terinstall di komputer Anda
- File model: `batik_model.tflite`

---

### Langkah 1 — Siapkan file model

Taruh file `batik_model.tflite` Anda di:
```
static/models/batik_model.tflite
```

Jika nama file model Anda berbeda, buka `app.py` baris 24 dan sesuaikan:
```python
MODEL_PATH = os.path.join(BASE_DIR, "static", "models", "NAMA_MODEL_ANDA.tflite")
```

---

### Langkah 2 — Sesuaikan label kelas

Buka `app.py` baris 29–44, ubah `CLASS_LABELS` sesuai urutan kelas training Anda:

```python
CLASS_LABELS = [
    "JawaBarat_GongSibolong",    # index 0
    "JawaBarat_MegaMendung",     # index 1
    # ... sesuaikan ...
]
```

Urutan harus **sama persis** dengan urutan folder dataset saat training.

---

### Langkah 3 — Buat repository GitHub

```bash
# Di terminal, masuk ke folder project
cd batik-tradjumas

# Inisialisasi git
git init
git add .
git commit -m "Initial commit - Batik Tradjumas Flask app"

# Buat repo di GitHub (lewat website github.com → New repository)
# Nama repo misalnya: batik-tradjumas
# Lalu hubungkan:
git remote add origin https://github.com/USERNAME_ANDA/batik-tradjumas.git
git branch -M main
git push -u origin main
```

> **Catatan:** File `.tflite` bisa besar. Jika > 100MB, gunakan Git LFS:
> ```bash
> git lfs install
> git lfs track "*.tflite"
> git add .gitattributes
> git add static/models/batik_model.tflite
> git commit -m "Add model with LFS"
> git push
> ```

---

### Langkah 4 — Deploy ke Railway

1. Buka https://railway.app → Login dengan GitHub
2. Klik **"New Project"**
3. Pilih **"Deploy from GitHub repo"**
4. Pilih repository `batik-tradjumas`
5. Railway otomatis deteksi Python dan install requirements
6. Tunggu build selesai (~3–5 menit pertama kali)
7. Setelah **"Deployed"**, klik **"Generate Domain"** untuk dapat URL publik

---

### Langkah 5 — Verifikasi

Setelah deploy, buka URL Railway Anda:

```
# Cek model berhasil dimuat:
https://NAMA-APP.up.railway.app/health

# Harusnya muncul:
{"model": true, "status": "ok"}

# Buka website:
https://NAMA-APP.up.railway.app/
https://NAMA-APP.up.railway.app/batiklens
```

---

### Langkah 6 — Test deteksi

1. Buka `/batiklens`
2. Klik **"Galeri"** → pilih foto kain batik
3. Klik **"Deteksi Motif"**
4. Hasil muncul dengan nama motif dan persentase keyakinan

---

## Jika Hasil Deteksi Selalu Salah

Kemungkinan masalah normalisasi. Buka `app.py` baris 49:

```python
# Ganti ini:
NORMALIZE = "div255"

# Menjadi ini (untuk MobileNetV2/EfficientNet):
NORMALIZE = "mobilenet"
```

Commit dan push → Railway otomatis redeploy.

---

## Update Kode / Model

Setiap kali ada perubahan, cukup:
```bash
git add .
git commit -m "Update: deskripsi perubahan"
git push
```
Railway akan otomatis redeploy dalam ~2 menit.

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Build gagal | Cek log Railway → biasanya versi package. Coba `tensorflow-cpu==2.15.0` |
| `/health` mengembalikan `"model": false` | Pastikan `batik_model.tflite` ada di `static/models/` |
| Deteksi lambat (~5–15 detik pertama) | Normal — model dimuat pertama kali (lazy load). Selanjutnya cepat |
| Error 503 | Model tidak ditemukan. Cek path di app.py |
| Error 500 | Lihat log Railway untuk detail error |
| Kamera tidak muncul | Browser butuh HTTPS. Railway otomatis HTTPS ✅ |
