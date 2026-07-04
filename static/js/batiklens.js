// batiklens.js — BatikLens Frontend
// Kirim gambar ke Flask endpoint /predict, tampilkan hasil deteksi motif batik.

document.addEventListener('DOMContentLoaded', function () {

  /* ── DOM refs ── */
  var stage       = document.getElementById('lensStage');
  var placeholder = document.getElementById('lensPlaceholder');
  var previewImg  = document.getElementById('lensPreview');
  var fileInput   = document.getElementById('fileInput');
  var galleryBtn  = document.getElementById('galleryBtn');
  var cameraBtn   = document.getElementById('cameraBtn');
  var detectBtn   = document.getElementById('detectBtn');
  var resetBtn    = document.getElementById('resetBtn');
  var statusEl    = document.getElementById('lensStatus');
  var resultEl    = document.getElementById('lensResult');
  var cameraModal = document.getElementById('cameraModal');
  var cameraVideo = document.getElementById('cameraVideo');
  var captureBtn  = document.getElementById('captureBtn');
  var closeCamBtn = document.getElementById('closeCameraBtn');

  var mediaStream    = null;
  var currentFile    = null;    // File dari input galeri
  var currentDataUrl = null;    // DataURL dari kamera

  // ── Status helper ──────────────────────────────────────────────────────
  function showStatus(msg, type) {
    statusEl.textContent = msg;
    statusEl.className = 'lens-status show ' + (type || '');
  }
  function clearStatus() {
    statusEl.className = 'lens-status';
    statusEl.textContent = '';
  }

  // ── Hasil deteksi ──────────────────────────────────────────────────────
  function showResult(data) {
    var pct = (data.confidence * 100).toFixed(1);

    // Bar chart top-3
    var bars = (data.top3 || []).map(function (item) {
      var p = (item.score * 100).toFixed(1);
      return (
        '<div class="result-bar-row">' +
          '<span class="result-bar-name">' + item.label + '</span>' +
          '<div class="result-bar-track">' +
            '<div class="result-bar-fill" style="width:0%"' +
              ' data-width="' + p + '"></div>' +
          '</div>' +
          '<span class="result-bar-pct">' + p + '%</span>' +
        '</div>'
      );
    }).join('');

    resultEl.innerHTML =
      '<div class="result-main">' +
        '<div class="result-main-left">' +
          '<span class="result-keterangan">Motif Terdeteksi</span>' +
          '<span class="result-label">' + data.label + '</span>' +
          (data.region ? '<span class="result-region">' + data.region + '</span>' : '') +
        '</div>' +
        '<span class="result-pct-badge">' + pct + '%</span>' +
      '</div>' +
      '<div class="result-bars">' +
        '<div class="result-bars-title">3 Kandidat Teratas</div>' +
        bars +
      '</div>';

    resultEl.classList.add('show');

    // Animasi bar setelah render
    requestAnimationFrame(function () {
      resultEl.querySelectorAll('.result-bar-fill').forEach(function (el) {
        el.style.width = el.getAttribute('data-width') + '%';
      });
    });
  }

  function clearResult() {
    resultEl.innerHTML = '';
    resultEl.classList.remove('show');
  }

  // ── Preview gambar ────────────────────────────────────────────────────
  function setPreview(dataUrl, file) {
    currentDataUrl = dataUrl;
    currentFile    = file || null;
    previewImg.src = dataUrl;
    previewImg.style.display = 'block';
    placeholder.style.display = 'none';
    stage.classList.add('has-image');
    detectBtn.disabled = false;
    resetBtn.hidden = false;
    clearResult();
    clearStatus();
  }

  function resetAll() {
    currentFile    = null;
    currentDataUrl = null;
    previewImg.removeAttribute('src');
    previewImg.style.display = 'none';
    placeholder.style.display = 'flex';
    stage.classList.remove('has-image', 'scanning');
    detectBtn.disabled = true;
    resetBtn.hidden = true;
    clearResult();
    clearStatus();
    fileInput.value = '';
  }

  // ── Galeri ────────────────────────────────────────────────────────────
  galleryBtn.addEventListener('click', function () { fileInput.click(); });

  fileInput.addEventListener('change', function (e) {
    var file = e.target.files && e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showStatus('File bukan gambar. Pilih JPG atau PNG.', 'error');
      return;
    }
    var reader = new FileReader();
    reader.onload = function (ev) { setPreview(ev.target.result, file); };
    reader.readAsDataURL(file);
  });

  // ── Kamera ───────────────────────────────────────────────────────────
  cameraBtn.addEventListener('click', async function () {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showStatus('Kamera tidak didukung di browser ini. Gunakan Galeri.', 'error');
      return;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }, audio: false
      });
      cameraVideo.srcObject = mediaStream;
      cameraModal.classList.add('open');
    } catch (err) {
      showStatus('Tidak bisa akses kamera: ' + err.message, 'error');
    }
  });

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(function (t) { t.stop(); });
      mediaStream = null;
    }
    cameraModal.classList.remove('open');
  }

  closeCamBtn.addEventListener('click', stopCamera);
  cameraModal.addEventListener('click', function (e) {
    if (e.target === cameraModal) stopCamera();
  });

  captureBtn.addEventListener('click', function () {
    var canvas = document.createElement('canvas');
    canvas.width  = cameraVideo.videoWidth  || 640;
    canvas.height = cameraVideo.videoHeight || 640;
    canvas.getContext('2d').drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
    setPreview(canvas.toDataURL('image/jpeg', 0.92), null);
    stopCamera();
  });

  // ── Reset ─────────────────────────────────────────────────────────────
  resetBtn.addEventListener('click', resetAll);

  // ── Kirim ke Flask /predict ───────────────────────────────────────────
  detectBtn.addEventListener('click', function () {
    if (!currentDataUrl && !currentFile) return;

    stage.classList.add('scanning');
    detectBtn.disabled = true;
    clearResult();
    showStatus('Mengirim gambar ke server…', 'loading');

    callPredict()
      .then(function (data) {
        stage.classList.remove('scanning');
        detectBtn.disabled = false;

        if (data.error) {
          showStatus('Error: ' + data.error, 'error');
          return;
        }
        clearStatus();
        showResult(data);
      })
      .catch(function (err) {
        stage.classList.remove('scanning');
        detectBtn.disabled = false;
        showStatus(
          'Gagal terhubung ke server. Pastikan Flask berjalan. (' + err.message + ')',
          'error'
        );
      });
  });

  function callPredict() {
    if (currentFile) {
      // Upload file → multipart (lebih efisien)
      var form = new FormData();
      form.append('image', currentFile);
      return fetch('/predict', { method: 'POST', body: form })
        .then(function (r) { return r.json(); });
    } else {
      // Kamera → JSON base64
      return fetch('/predict', {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ image: currentDataUrl })
      }).then(function (r) { return r.json(); });
    }
  }

  // ── Init ──────────────────────────────────────────────────────────────
  resetAll();
});
