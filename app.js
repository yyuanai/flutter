/* ══════════════════════════════════════════════════════════
   Flutter — Butterfly Identifier  |  app.js
   Full TensorFlow.js integration + UI logic
══════════════════════════════════════════════════════════ */

'use strict';

/* ─── STATE ─── */
const State = {
  model:       null,
  labels:      [],
  stream:      null,
  facingMode:  'environment',
  torchOn:     false,
  currentPage: 'home',
  history:     JSON.parse(localStorage.getItem('bf_history') || '[]'),
};

/* ─── FUN FACTS (fallback per common butterfly names) ─── */
const FUN_FACTS = [
  "Butterflies taste with their feet — they have chemoreceptors on their tarsi!",
  "A group of butterflies is called a 'kaleidoscope' or 'flutter'.",
  "Butterflies can see ultraviolet light patterns invisible to the human eye.",
  "Some butterflies migrate thousands of miles each season.",
  "Butterfly wings are actually transparent — the colours come from tiny reflective scales.",
  "They can only fly if their body temperature is above 29°C.",
  "The largest butterfly wingspan reaches nearly 30 cm in the Queen Alexandra's Birdwing.",
  "Butterflies drink from mud puddles to absorb minerals — a behaviour called 'puddling'.",
  "Most adult butterflies live only 2–4 weeks, though some overwinter for 6–8 months.",
  "Monarch butterflies navigate using both the sun and Earth's magnetic field.",
];

/* ════════════════════════════════════════════════════════
   BOOT — load model + splash dismiss
════════════════════════════════════════════════════════ */
async function boot() {
  try {
    await loadModel();
  } catch (e) {
    console.warn('Model load failed:', e);
    setModelStatus('error');
  }

  // Wait for loader animation then dismiss splash
  await delay(2800);
  dismissSplash();
}

async function loadModel() {
  // Load metadata first
  try {
    const metaResp = await fetch('metadata.json');
    if (metaResp.ok) {
      const meta = await metaResp.json();
      State.labels = meta.labels || meta.classes || Object.values(meta) || [];
    }
  } catch (_) {
    console.warn('metadata.json not found — using fallback labels');
    State.labels = generateFallbackLabels();
  }

  // Load TF model
  try {
    State.model = await tf.loadLayersModel('model.json');
    console.log('✅ Model loaded successfully');
    setModelStatus('ready');
    showToast('🦋 AI model ready!');
  } catch (e) {
    console.warn('model.json not found — running in demo mode:', e);
    State.model = null;          // demo mode
    setModelStatus('error');
    showToast('⚠️ Demo mode — no model found');
    if (State.labels.length === 0) State.labels = generateFallbackLabels();
  }
}

function generateFallbackLabels() {
  return [
    'Monarch', 'Blue Morpho', 'Painted Lady', 'Swallowtail',
    'Red Admiral', 'Cabbage White', 'Common Brimstone',
    'Peacock', 'Small Tortoiseshell', 'Orange-tip',
    'Purple Emperor', 'Clouded Yellow', 'Comma',
    'Grayling', 'Marbled White', 'Meadow Brown',
  ];
}

function dismissSplash() {
  const splash = document.getElementById('splash');
  const app    = document.getElementById('app');
  splash.classList.add('fade-out');
  app.classList.remove('hidden');
  setTimeout(() => splash.style.display = 'none', 700);
  renderHomeHistory();
}

/* ════════════════════════════════════════════════════════
   PREDICTION ENGINE
════════════════════════════════════════════════════════ */
async function predict(imageEl) {
  if (!State.model) {
    // Demo mode: random-ish scores
    return demoPredict();
  }

  return tf.tidy(() => {
    const INPUT_SIZE = 224; // standard MobileNet / TeachableMachine size

    let tensor = tf.browser.fromPixels(imageEl)
      .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
      .toFloat()
      .div(255.0)
      .expandDims(0);

    const output = State.model.predict(tensor);
    const scores = Array.from(output.dataSync());

    return buildTopK(scores, 3);
  });
}

function demoPredict() {
  // Simulate predictions for demo/test
  const n = State.labels.length;
  let scores = Array.from({ length: n }, () => Math.random());
  const sum = scores.reduce((a, b) => a + b, 0);
  scores = scores.map(s => s / sum);
  return buildTopK(scores, 3);
}

function buildTopK(scores, k) {
  const indexed = scores.map((s, i) => ({ score: s, label: State.labels[i] || `Species ${i + 1}` }));
  indexed.sort((a, b) => b.score - a.score);
  return indexed.slice(0, k);
}

/* ════════════════════════════════════════════════════════
   CAMERA PAGE
════════════════════════════════════════════════════════ */
async function startCamera() {
  stopCamera();
  const video = document.getElementById('camera-video');
  try {
    State.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: State.facingMode, width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    video.srcObject = State.stream;
  } catch (e) {
    showToast('📷 Camera access denied');
    console.error(e);
  }
}

function stopCamera() {
  if (State.stream) {
    State.stream.getTracks().forEach(t => t.stop());
    State.stream = null;
  }
}

async function captureAndPredict() {
  const video  = document.getElementById('camera-video');
  const canvas = document.getElementById('camera-canvas');
  if (!video.srcObject) { showToast('📷 Start camera first'); return; }

  // Start scan animation
  const scanLine = document.getElementById('scan-line');
  scanLine.classList.add('scanning');

  // Show analyzing overlay on camera frame
  const frame = document.querySelector('.camera-frame');
  const overlay = showAnalyzingOverlay(frame);

  // Draw frame to canvas
  canvas.width  = video.videoWidth  || 224;
  canvas.height = video.videoHeight || 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL('image/jpeg', 0.9);

  // Create temp image for TF
  const img = new Image();
  img.src = dataURL;
  await new Promise(r => { img.onload = r; });

  const results = await predict(img);

  scanLine.classList.remove('scanning');
  removeAnalyzingOverlay(overlay);

  renderResult('camera-result', dataURL, results);
}

document.getElementById('cam-switch-btn').addEventListener('click', () => {
  State.facingMode = State.facingMode === 'environment' ? 'user' : 'environment';
  if (State.stream) startCamera();
});

document.getElementById('cam-torch-btn').addEventListener('click', async () => {
  if (!State.stream) return;
  const track = State.stream.getVideoTracks()[0];
  if (!track) return;
  State.torchOn = !State.torchOn;
  try {
    await track.applyConstraints({ advanced: [{ torch: State.torchOn }] });
    document.getElementById('cam-torch-btn').style.opacity = State.torchOn ? '0.5' : '1';
  } catch (_) {
    showToast('Torch not available');
  }
});

document.getElementById('capture-btn').addEventListener('click', captureAndPredict);

/* ════════════════════════════════════════════════════════
   UPLOAD PAGE
════════════════════════════════════════════════════════ */
const uploadZone  = document.getElementById('upload-zone');
const fileInput   = document.getElementById('file-input');
const previewWrap = document.getElementById('upload-preview-wrap');
const previewImg  = document.getElementById('upload-preview');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

document.getElementById('clear-upload-btn').addEventListener('click', () => {
  fileInput.value = '';
  previewWrap.classList.add('hidden');
  uploadZone.classList.remove('hidden');
  document.getElementById('upload-result').classList.add('hidden');
});

async function handleFile(file) {
  if (!file.type.startsWith('image/')) { showToast('⚠️ Please select an image'); return; }

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.classList.remove('hidden');
  uploadZone.classList.add('hidden');

  // Wait for image to load
  await new Promise(r => { previewImg.onload = r; });

  // Show overlay
  const overlay = showAnalyzingOverlay(previewWrap);
  const results = await predict(previewImg);
  removeAnalyzingOverlay(overlay);

  renderResult('upload-result', previewImg.src, results);
}

/* ════════════════════════════════════════════════════════
   RESULT CARD RENDERER
════════════════════════════════════════════════════════ */
function renderResult(containerId, imageURL, results) {
  const container = document.getElementById(containerId);
  container.classList.remove('hidden');

  const top    = results[0];
  const pct    = (top.score * 100).toFixed(1);
  const fact   = FUN_FACTS[Math.floor(Math.random() * FUN_FACTS.length)];
  const saveId = `save_${Date.now()}`;

  container.innerHTML = `
    <div class="result-top">
      <img class="result-thumb" src="${imageURL}" alt="${top.label}" />
      <div class="result-meta">
        <div class="result-label">🦋 Identified Species</div>
        <div class="result-species">${top.label}</div>
        <div class="result-confidence">${pct}% confidence</div>
      </div>
    </div>

    <div class="progress-wrap">
      <div class="progress-label">
        <span>Confidence</span><span>${pct}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-bar" id="pb_${saveId}" style="width:0%"></div>
      </div>
    </div>

    ${results.length > 1 ? `
    <div class="top-predictions">
      <div class="tp-label">Top Predictions</div>
      ${results.map((r, i) => `
        <div class="tp-item">
          <div class="tp-rank">${i + 1}</div>
          <div class="tp-name">${r.label}</div>
          <div class="tp-pct">${(r.score * 100).toFixed(1)}%</div>
        </div>
      `).join('')}
    </div>
    ` : ''}

    <div class="fun-fact-card">
      <div class="ff-header">
        <span class="ff-icon">✨</span>
        <span class="ff-title">Butterfly Fact</span>
      </div>
      <p class="ff-text">${fact}</p>
    </div>

    <button class="save-btn" id="${saveId}" onclick="saveToHistory('${saveId}', \`${encodeURIComponent(imageURL)}\`, \`${encodeURIComponent(top.label)}\`, '${pct}')">
      🌸 Save to My Garden
    </button>
  `;

  // Animate progress bar
  requestAnimationFrame(() => {
    setTimeout(() => {
      const bar = document.getElementById(`pb_${saveId}`);
      if (bar) bar.style.width = `${pct}%`;
    }, 100);
  });

  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ════════════════════════════════════════════════════════
   HISTORY
════════════════════════════════════════════════════════ */
function saveToHistory(btnId, encodedURL, encodedSpecies, pct) {
  const btn     = document.getElementById(btnId);
  const imageURL = decodeURIComponent(encodedURL);
  const species  = decodeURIComponent(encodedSpecies);

  // Save as small thumbnail
  const canvas = document.createElement('canvas');
  canvas.width = 100; canvas.height = 100;
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = imageURL;
  img.onload = () => {
    ctx.drawImage(img, 0, 0, 100, 100);
    const thumb = canvas.toDataURL('image/jpeg', 0.6);
    const entry = {
      id:      Date.now(),
      thumb,
      species,
      pct,
      date:    new Date().toLocaleString(),
    };
    State.history.unshift(entry);
    if (State.history.length > 50) State.history.pop();
    localStorage.setItem('bf_history', JSON.stringify(State.history));
    showToast('🌸 Saved to your garden!');
    if (btn) { btn.textContent = '✅ Saved!'; btn.disabled = true; }
    renderHistoryPage();
    renderHomeHistory();
  };
  img.onerror = () => {
    // No canvas thumb available, save without
    const entry = { id: Date.now(), thumb: null, species, pct, date: new Date().toLocaleString() };
    State.history.unshift(entry);
    localStorage.setItem('bf_history', JSON.stringify(State.history));
    showToast('🌸 Saved to your garden!');
    if (btn) { btn.textContent = '✅ Saved!'; btn.disabled = true; }
    renderHistoryPage();
    renderHomeHistory();
  };
}

function renderHistoryPage() {
  const list = document.getElementById('history-list');
  if (State.history.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-emoji">🌿</div>
        <h3>Your garden is empty</h3>
        <p>Identified butterflies will appear here</p>
        <button class="cta-btn" onclick="navigateTo('camera')">Start Scanning</button>
      </div>`;
    return;
  }

  list.innerHTML = State.history.map(item => `
    <div class="history-item" id="hi_${item.id}">
      ${item.thumb
        ? `<img class="hi-thumb" src="${item.thumb}" alt="${item.species}" />`
        : `<div class="hi-thumb" style="background:linear-gradient(135deg,#e9d5ff,#fce7f3);display:flex;align-items:center;justify-content:center;font-size:24px;">🦋</div>`
      }
      <div class="hi-info">
        <div class="hi-species">${item.species}</div>
        <div class="hi-confidence">${item.pct}% confident</div>
        <div class="hi-date">📅 ${item.date}</div>
      </div>
      <button class="hi-delete" onclick="deleteHistoryItem(${item.id})">🗑️</button>
    </div>
  `).join('');
}

function deleteHistoryItem(id) {
  State.history = State.history.filter(h => h.id !== id);
  localStorage.setItem('bf_history', JSON.stringify(State.history));
  const el = document.getElementById(`hi_${id}`);
  if (el) { el.style.opacity = '0'; el.style.transform = 'scale(0.9)'; el.style.transition = '0.25s'; setTimeout(() => renderHistoryPage(), 250); }
  renderHomeHistory();
}

document.getElementById('clear-all-btn').addEventListener('click', () => {
  if (State.history.length === 0) return;
  if (!confirm('Clear all scans from your garden?')) return;
  State.history = [];
  localStorage.removeItem('bf_history');
  renderHistoryPage();
  renderHomeHistory();
  showToast('🌿 Garden cleared');
});

function renderHomeHistory() {
  const wrap = document.getElementById('home-history-preview');
  if (State.history.length === 0) {
    wrap.innerHTML = `
      <div class="empty-state-sm">
        <span>🌿</span><p>No scans yet — go explore!</p>
      </div>`;
    return;
  }
  wrap.innerHTML = State.history.slice(0, 3).map(item => `
    <div class="home-hi-card">
      ${item.thumb
        ? `<img class="home-hi-thumb" src="${item.thumb}" alt="${item.species}" />`
        : `<div class="home-hi-thumb" style="background:linear-gradient(135deg,#e9d5ff,#fce7f3);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;width:48px;height:48px;">🦋</div>`
      }
      <div>
        <div class="home-hi-name">${item.species}</div>
        <div class="home-hi-pct">${item.pct}% · ${item.date}</div>
      </div>
    </div>
  `).join('');
}

/* ════════════════════════════════════════════════════════
   NAVIGATION
════════════════════════════════════════════════════════ */
function navigateTo(page) {
  if (page === State.currentPage) return;

  // Stop camera when leaving camera page
  if (State.currentPage === 'camera') stopCamera();

  // Deactivate current page
  document.getElementById(`page-${State.currentPage}`).classList.remove('active');
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));

  State.currentPage = page;

  // Activate new page
  document.getElementById(`page-${page}`).classList.add('active');
  const navBtn = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (navBtn) navBtn.classList.add('active');

  // Page-specific init
  if (page === 'camera') startCamera();
  if (page === 'history') renderHistoryPage();
}

// Nav dock buttons
document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => navigateTo(btn.dataset.page));
});

/* ════════════════════════════════════════════════════════
   HELPERS
════════════════════════════════════════════════════════ */
function setModelStatus(status) {
  const dot = document.getElementById('model-dot');
  dot.className = `status-dot ${status}`;
}

let toastTimer;
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  requestAnimationFrame(() => t.classList.add('show'));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => t.classList.add('hidden'), 300);
  }, 2500);
}

function showAnalyzingOverlay(parent) {
  const overlay = document.createElement('div');
  overlay.className = 'analyzing-overlay';
  overlay.innerHTML = `
    <div class="analyzing-spinner">🦋</div>
    <div class="analyzing-text">Identifying species…</div>
  `;
  parent.style.position = 'relative';
  parent.appendChild(overlay);
  return overlay;
}

function removeAnalyzingOverlay(overlay) {
  if (overlay && overlay.parentNode) {
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.2s';
    setTimeout(() => overlay.remove(), 200);
  }
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ════════════════════════════════════════════════════════
   INIT
════════════════════════════════════════════════════════ */
boot();

/* ─── Service Worker registration ─── */
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('service-worker.js')
      .then(reg => console.log('SW registered:', reg.scope))
      .catch(err => console.warn('SW not found:', err));
  });
}