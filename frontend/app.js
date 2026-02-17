const state = {
  sessionId: null,
  ws: null,
  running: false,
  losses: [],
  trace: [],
  lastEvent: null,
};

const els = {
  status: document.getElementById("status-pill"),
  meta: document.getElementById("meta"),
  stepStats: document.getElementById("step-stats"),
  lossCanvas: document.getElementById("loss-chart"),
  heatmapHead: document.getElementById("heatmap-head"),
  heatmap: document.getElementById("attention-heatmap"),
  probBars: document.getElementById("prob-bars"),
  tokenTrail: document.getElementById("token-trail"),
  progressTrack: document.getElementById("progress-fill"),
  progressLabel: document.getElementById("progress-label"),
  samples: document.getElementById("samples"),
  startBtn: document.getElementById("start-btn"),
  stepBtn: document.getElementById("step-btn"),
  runBtn: document.getElementById("run-btn"),
  pauseBtn: document.getElementById("pause-btn"),
  sampleBtn: document.getElementById("sample-btn"),
  resetBtn: document.getElementById("reset-btn"),
};

const graphNodes = [
  { id: "embed", label: "Embedding", x: 20, y: 42 },
  { id: "norm", label: "RMSNorm", x: 150, y: 42 },
  { id: "qkv", label: "QKV", x: 280, y: 42 },
  { id: "attn", label: "Attention", x: 410, y: 42 },
  { id: "mlp", label: "MLP", x: 555, y: 42 },
  { id: "logits", label: "Logits", x: 685, y: 42 },
  { id: "loss", label: "Loss", x: 810, y: 42 },
  { id: "backprop", label: "Backprop", x: 260, y: 122 },
  { id: "adam", label: "Adam Update", x: 530, y: 122 },
];

initGraph();
wireActions();
render();

function wireActions() {
  els.startBtn.addEventListener("click", startSession);
  els.stepBtn.addEventListener("click", stepSession);
  els.runBtn.addEventListener("click", runSession);
  els.pauseBtn.addEventListener("click", pauseSession);
  els.sampleBtn.addEventListener("click", sampleSession);
  els.resetBtn.addEventListener("click", resetSession);
  window.addEventListener("keydown", handleKeyboard);
}

function cfgValue(id) {
  const el = document.getElementById(id);
  return Number(el.value);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

async function startSession() {
  try {
    const payload = {
      n_embd: cfgValue("n-embd"),
      n_head: cfgValue("n-head"),
      n_layer: cfgValue("n-layer"),
      block_size: cfgValue("block-size"),
      learning_rate: cfgValue("learning-rate"),
      num_steps: cfgValue("num-steps"),
      temperature: cfgValue("temperature"),
      seed: 42,
    };
    const res = await api("/api/session/start", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    state.sessionId = res.session_id;
    state.losses = [];
    state.trace = [];
    state.lastEvent = null;
    els.samples.innerHTML = "";
    bindSocket(res.session_id);

    setButtonsEnabled(true);
    renderMeta(res.metadata);
    setStatus("Session Ready", false);
    render();
  } catch (err) {
    alert(`Failed to start session: ${err.message}`);
  }
}

async function stepSession() {
  if (!state.sessionId) {
    return;
  }
  try {
    const event = await api(`/api/session/${state.sessionId}/step`, { method: "POST" });
    onTrainEvent(event);
  } catch (err) {
    alert(`Step failed: ${err.message}`);
  }
}

async function runSession() {
  if (!state.sessionId) {
    return;
  }
  try {
    const payload = {
      steps: cfgValue("run-steps"),
      delay_ms: cfgValue("run-delay"),
    };
    await api(`/api/session/${state.sessionId}/run`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setStatus("Running", true);
  } catch (err) {
    alert(`Run failed: ${err.message}`);
  }
}

async function pauseSession() {
  if (!state.sessionId) {
    return;
  }
  try {
    await api(`/api/session/${state.sessionId}/pause`, { method: "POST" });
    setStatus("Paused", false);
  } catch (err) {
    alert(`Pause failed: ${err.message}`);
  }
}

async function sampleSession() {
  if (!state.sessionId) {
    return;
  }
  try {
    const payload = {
      num_samples: 10,
      temperature: cfgValue("temperature"),
    };
    const res = await api(`/api/session/${state.sessionId}/sample`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderSamples(res.samples);
  } catch (err) {
    alert(`Sample failed: ${err.message}`);
  }
}

function handleKeyboard(event) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
    return;
  }

  const key = event.key.toLowerCase();
  if (!state.sessionId && !["n", "enter"].includes(key)) {
    return;
  }

  if (key === "enter") {
    event.preventDefault();
    if (!state.sessionId) {
      startSession();
    }
    return;
  }

  if (key === "n") {
    event.preventDefault();
    startSession();
    return;
  }
  if (key === "s") {
    event.preventDefault();
    stepSession();
    return;
  }
  if (key === "r") {
    event.preventDefault();
    runSession();
    return;
  }
  if (key === "p") {
    event.preventDefault();
    pauseSession();
    return;
  }
  if (key === "c") {
    event.preventDefault();
    sampleSession();
  }
}

async function resetSession() {
  if (!state.sessionId) {
    return;
  }
  try {
    const res = await api(`/api/session/${state.sessionId}/reset`, { method: "POST" });
    state.losses = [];
    state.trace = [];
    state.lastEvent = null;
    render();
    renderSamples([]);
    renderMeta(res.metadata);
  } catch (err) {
    alert(`Reset failed: ${err.message}`);
  }
}

function bindSocket(sessionId) {
  if (state.ws) {
    state.ws.close();
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);
  state.ws = ws;

  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
  if (data.type === "train_step") {
      onTrainEvent(data);
    } else if (data.type === "run_status") {
      setStatus(data.running ? "Running" : "Session Ready", data.running);
    } else if (data.type === "samples") {
      renderSamples(data.samples);
    } else if (data.type === "session") {
      renderMeta(data.metadata);
    } else if (data.type === "run_status") {
      setStatus(data.running ? "Running" : "Session Ready", data.running);
    }
  };

  ws.onclose = () => {
    if (state.sessionId) {
      setStatus("Disconnected", false);
    }
  };
}

function onTrainEvent(event) {
  state.lastEvent = event;
  state.losses.push(event.loss);
  state.trace.push(event);
  if (state.losses.length > 500) {
    state.losses.shift();
  }
  if (state.trace.length > 500) {
    state.trace.shift();
  }

  pulseGraph([
    "embed",
    "norm",
    "qkv",
    "attn",
    "mlp",
    "logits",
    "loss",
    "backprop",
    "adam",
  ]);

  render();
}

function setButtonsEnabled(enabled) {
  els.stepBtn.disabled = !enabled;
  els.runBtn.disabled = !enabled;
  els.pauseBtn.disabled = !enabled;
  els.sampleBtn.disabled = !enabled;
}

function setStatus(text, running) {
  state.running = running;
  els.status.textContent = text;
  els.status.classList.toggle("running", running);
}

function renderMeta(metadata) {
  els.meta.textContent = JSON.stringify(metadata, null, 2);
}

function render() {
  renderLossChart();
  renderStepStats();
  renderProgress();
  renderAttention();
  renderProbs();
  renderTrail();
}

function renderStepStats() {
  if (!state.lastEvent) {
    els.stepStats.textContent = "No training step yet.";
    return;
  }
  const e = state.lastEvent;
  const pct = (e.progress ?? 0) * 100;
  els.stepStats.textContent = `step ${e.step}/${e.num_steps} | loss ${e.loss.toFixed(4)} | grad norm ${e.grad_norm.toFixed(4)} | lr ${e.learning_rate.toExponential(2)} | progress ${pct.toFixed(1)}%`;
}

function renderProgress() {
  const latest = state.lastEvent;
  const progress = latest?.progress ?? 0;
  const pct = Math.min(1, Math.max(0, progress));
  els.progressTrack.style.width = `${(pct * 100).toFixed(1)}%`;
  const remaining = latest?.steps_remaining ?? (latest?.num_steps ? latest?.num_steps - latest?.step : "n/a");
  els.progressLabel.textContent = `${latest ? `${(pct * 100).toFixed(1)}%` : "0.0%"}${typeof remaining === "number" ? ` | remaining ${remaining}` : ""}`;
}

function renderLossChart() {
  const ctx = els.lossCanvas.getContext("2d");
  const width = els.lossCanvas.width;
  const height = els.lossCanvas.height;

  ctx.clearRect(0, 0, width, height);

  if (state.losses.length < 2) {
    ctx.fillStyle = "#5a6c80";
    ctx.font = "14px sans-serif";
    ctx.fillText("Run a few steps to see loss dynamics.", 20, 28);
    return;
  }

  const min = Math.min(...state.losses);
  const max = Math.max(...state.losses);
  const range = Math.max(1e-6, max - min);

  ctx.strokeStyle = "rgba(20,35,55,0.15)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = 20 + ((height - 40) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(15, y);
    ctx.lineTo(width - 10, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#ff6b3d";
  ctx.lineWidth = 2.2;
  ctx.beginPath();

  state.losses.forEach((val, i) => {
    const x = 15 + ((width - 25) * i) / (state.losses.length - 1);
    const y = 20 + ((height - 40) * (max - val)) / range;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  const latest = state.losses[state.losses.length - 1];
  ctx.fillStyle = "#17324f";
  ctx.font = "12px sans-serif";
  ctx.fillText(`min ${min.toFixed(3)} max ${max.toFixed(3)} latest ${latest.toFixed(3)}`, 18, height - 10);
}

function renderAttention() {
  const event = state.lastEvent;
  if (!event) {
    els.heatmapHead.textContent = "No token yet";
    els.heatmap.innerHTML = "";
    return;
  }

  const tokenEvent = event.last_token_event;
  const head = tokenEvent.attention?.[0]?.heads?.[0];
  if (!head) {
    els.heatmapHead.textContent = "Attention data unavailable";
    els.heatmap.innerHTML = "";
    return;
  }

  els.heatmapHead.textContent = `Layer 0 Head 0 | position ${tokenEvent.pos} | input '${tokenEvent.input_token}' -> target '${tokenEvent.target_token}'`;
  els.heatmap.innerHTML = "";

  const rows = Math.max(8, head.weights.length);
  const padded = [...head.weights];
  while (padded.length < rows) {
    padded.push(0);
  }

  padded.forEach((w, idx) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    const c = Math.min(1, Math.max(0, w));
    const alpha = 0.1 + c * 0.9;
    cell.style.background = `rgba(3, 134, 208, ${alpha.toFixed(3)})`;
    cell.style.color = c > 0.55 ? "#fff" : "#0f2e4a";
    cell.textContent = idx < head.weights.length ? w.toFixed(2) : "-";
    els.heatmap.appendChild(cell);
  });
}

function renderProbs() {
  const event = state.lastEvent;
  if (!event) {
    els.probBars.innerHTML = "";
    return;
  }

  const top = event.last_token_event?.top_probs || [];
  els.probBars.innerHTML = "";

  top.forEach((row) => {
    const r = document.createElement("div");
    r.className = "bar-row";
    const pct = row.prob * 100;
    r.innerHTML = `
      <div><strong>${escapeHtml(row.token)}</strong></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(2)}%"></div></div>
      <div>${pct.toFixed(2)}%</div>
    `;
    els.probBars.appendChild(r);
  });
}

function renderTrail() {
  const event = state.lastEvent;
  if (!event) {
    els.tokenTrail.textContent = "";
    return;
  }
  const tokens = event.sequence.tokens.join(" -> ");
  els.tokenTrail.textContent = `Current sequence: ${tokens}`;
}

function renderSamples(samples) {
  els.samples.innerHTML = "";
  samples.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s || "<empty>";
    els.samples.appendChild(li);
  });
}

function initGraph() {
  const svg = document.getElementById("graph");
  const nodesLayer = svg.querySelector(".nodes");
  const edgesLayer = svg.querySelector(".edges");

  graphNodes.forEach((node) => {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.dataset.nodeId = node.id;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", node.x);
    rect.setAttribute("y", node.y);
    rect.setAttribute("width", "108");
    rect.setAttribute("height", "36");

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", String(node.x + 54));
    text.setAttribute("y", String(node.y + 22));
    text.setAttribute("text-anchor", "middle");
    text.textContent = node.label;

    g.appendChild(rect);
    g.appendChild(text);
    nodesLayer.appendChild(g);
  });

  const links = [
    ["embed", "norm"],
    ["norm", "qkv"],
    ["qkv", "attn"],
    ["attn", "mlp"],
    ["mlp", "logits"],
    ["logits", "loss"],
    ["loss", "backprop"],
    ["backprop", "adam"],
  ];

  links.forEach(([a, b]) => {
    const from = graphNodes.find((n) => n.id === a);
    const to = graphNodes.find((n) => n.id === b);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const x1 = from.x + 108;
    const y1 = from.y + 18;
    const x2 = to.x;
    const y2 = to.y + 18;
    path.setAttribute("d", `M${x1},${y1} C${x1 + 30},${y1} ${x2 - 30},${y2} ${x2},${y2}`);
    edgesLayer.appendChild(path);
  });
}

function pulseGraph(ids) {
  const nodes = [...document.querySelectorAll(".nodes g")];
  nodes.forEach((n) => n.classList.remove("active"));

  ids.forEach((id, i) => {
    setTimeout(() => {
      const node = document.querySelector(`.nodes g[data-node-id='${id}']`);
      if (!node) {
        return;
      }
      nodes.forEach((n) => n.classList.remove("active"));
      node.classList.add("active");
    }, i * 85);
  });
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
