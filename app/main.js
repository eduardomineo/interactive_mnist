const GRID_SIZE = 28;
const PIXEL_COUNT = GRID_SIZE * GRID_SIZE;
const VIRTUAL_WIDTH = 960;
const VIRTUAL_HEIGHT = 900;
const CANVAS_ASPECT = VIRTUAL_WIDTH / VIRTUAL_HEIGHT;
const DPR = window.devicePixelRatio || 1;

const canvas = document.getElementById("mnistCanvas");
const ctx = canvas.getContext("2d");
const workspace = document.querySelector(".workspace");
const clearButton = document.getElementById("clearButton");
const statusEl = document.getElementById("status");

const pixels = new Float32Array(PIXEL_COUNT);
const hidden1 = new Float32Array(25);
const hidden2 = new Float32Array(25);
const output = new Float32Array(10);

let model = null;
let activationModel = null;
let drawing = false;
let inferenceTimer = 0;
let inferenceRunning = false;
let inferenceDirty = false;
let predictedDigit = -1;

const layout = {
  outputY: 62,
  hidden1Y: 190,
  hidden2Y: 310,
  gridY: 430,
};

function resizeCanvas() {
  const availableWidth = workspace.clientWidth;
  const availableHeight = workspace.clientHeight;
  const cssWidth = Math.floor(Math.min(availableWidth, availableHeight * CANVAS_ASPECT));
  const cssHeight = Math.round(cssWidth / CANVAS_ASPECT);

  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * DPR);
  canvas.height = Math.round(cssHeight * DPR);
  ctx.setTransform(
    (cssWidth / VIRTUAL_WIDTH) * DPR,
    0,
    0,
    (cssHeight / VIRTUAL_HEIGHT) * DPR,
    0,
    0
  );
  render();
}

function getCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * VIRTUAL_WIDTH,
    y: ((event.clientY - rect.top) / rect.height) * VIRTUAL_HEIGHT,
  };
}

function getGridRect() {
  const width = VIRTUAL_WIDTH;
  const size = Math.min(width - 48, 420);
  return {
    x: (width - size) / 2,
    y: Math.min(layout.gridY, VIRTUAL_HEIGHT - size - 28),
    size,
    cell: size / GRID_SIZE,
  };
}

function paintAt(point) {
  const grid = getGridRect();
  const col = Math.floor((point.x - grid.x) / grid.cell);
  const row = Math.floor((point.y - grid.y) / grid.cell);

  if (col < 0 || col >= GRID_SIZE || row < 0 || row >= GRID_SIZE) {
    return;
  }

  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const x = col + dx;
      const y = row + dy;
      if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) {
        continue;
      }
      const distance = Math.hypot(dx, dy);
      const strength = distance === 0 ? 1 : 0.38;
      const index = y * GRID_SIZE + x;
      pixels[index] = Math.max(pixels[index], strength);
    }
  }

  scheduleInference();
  render();
}

function scheduleInference() {
  inferenceDirty = true;
  if (inferenceRunning || inferenceTimer) {
    return;
  }
  inferenceTimer = window.setTimeout(runInference, 40);
}

async function loadModel() {
  try {
    model = await tf.loadLayersModel("model/model.json");
    const h1 = model.getLayer("hidden_1").output;
    const h2 = model.getLayer("hidden_2").output;
    activationModel = tf.model({
      inputs: model.inputs,
      outputs: [h1, h2, model.outputs[0]],
    });
    statusEl.textContent = "Draw a digit";
    await runInference();
  } catch (error) {
    statusEl.textContent = "Model not found. Run the training/export command first.";
  }
}

async function runInference() {
  inferenceTimer = 0;
  if (!activationModel) {
    return;
  }
  if (!inferenceDirty || inferenceRunning) {
    return;
  }

  inferenceDirty = false;
  inferenceRunning = true;
  let input = null;
  let h1Tensor = null;
  let h2Tensor = null;
  let outputTensor = null;
  try {
    input = tf.tensor2d(pixels, [1, PIXEL_COUNT]);
    [h1Tensor, h2Tensor, outputTensor] = activationModel.predict(input);

    const [h1Values, h2Values, outputValues] = await Promise.all([
      h1Tensor.data(),
      h2Tensor.data(),
      outputTensor.data(),
    ]);

    hidden1.set(h1Values);
    hidden2.set(h2Values);
    output.set(outputValues);
    predictedDigit = output.indexOf(Math.max(...output));

    statusEl.textContent = predictedDigit >= 0 ? `Prediction: ${predictedDigit}` : "Draw a digit";
    render();
  } finally {
    tf.dispose([input, h1Tensor, h2Tensor, outputTensor].filter(Boolean));
    inferenceRunning = false;
    if (inferenceDirty) {
      scheduleInference();
    }
  }
}

function clearDrawing() {
  pixels.fill(0);
  hidden1.fill(0);
  hidden2.fill(0);
  output.fill(0);
  predictedDigit = -1;
  statusEl.textContent = activationModel ? "Draw a digit" : statusEl.textContent;
  scheduleInference();
  render();
}

function neuronPositions(count, y, size, gap) {
  const totalWidth = count * size + (count - 1) * gap;
  const startX = (VIRTUAL_WIDTH - totalWidth) / 2;
  return Array.from({ length: count }, (_, index) => ({
    x: startX + index * (size + gap),
    y,
    size,
  }));
}

function drawConnections(from, to, color, offset) {
  ctx.save();
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.22;
  ctx.strokeStyle = color;

  for (let i = 0; i < from.length; i += 1) {
    for (let j = 0; j < to.length; j += 1) {
      if ((i * 7 + j * 11 + offset) % 9 !== 0) {
        continue;
      }
      const a = from[i];
      const b = to[j];
      ctx.beginPath();
      ctx.moveTo(a.x + a.size / 2, a.y + a.size);
      ctx.lineTo(b.x + b.size / 2, b.y);
      ctx.stroke();
    }
  }

  ctx.restore();
}

function drawOutputSquares(positions) {
  ctx.save();
  ctx.font = "700 22px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  positions.forEach((pos, digit) => {
    const probability = output[digit] || 0;
    ctx.fillStyle = "#16181d";
    ctx.fillText(String(digit), pos.x + pos.size / 2, pos.y - 24);

    ctx.lineWidth = digit === predictedDigit ? 4 : 2;
    ctx.strokeStyle = digit === predictedDigit ? "#0f766e" : "#20232a";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(pos.x, pos.y, pos.size, pos.size);
    ctx.strokeRect(pos.x, pos.y, pos.size, pos.size);

    const fillHeight = pos.size * probability;
    ctx.fillStyle = digit === predictedDigit ? "#0f766e" : "#1f2937";
    ctx.fillRect(pos.x, pos.y + pos.size - fillHeight, pos.size, fillHeight);
  });

  ctx.restore();
}

function drawHiddenSquares(positions, activations) {
  ctx.save();
  positions.forEach((pos, index) => {
    const normalized = (activations[index] + 1) / 2;
    const clamped = Math.max(0, Math.min(1, normalized));
    const fillHeight = pos.size * clamped;

    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#20232a";
    ctx.lineWidth = 2;
    ctx.fillRect(pos.x, pos.y, pos.size, pos.size);
    ctx.strokeRect(pos.x, pos.y, pos.size, pos.size);

    ctx.fillStyle = activations[index] >= 0 ? "#111827" : "#9ca3af";
    ctx.fillRect(pos.x, pos.y + pos.size - fillHeight, pos.size, fillHeight);
  });
  ctx.restore();
}

function drawInputGrid() {
  const grid = getGridRect();

  ctx.save();
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const value = pixels[row * GRID_SIZE + col];
      const shade = Math.round(236 - value * 220);
      ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade + Math.round((1 - value) * 6)})`;
      ctx.fillRect(
        grid.x + col * grid.cell,
        grid.y + row * grid.cell,
        grid.cell - 1,
        grid.cell - 1
      );
    }
  }

  ctx.strokeStyle = "#cdd2db";
  ctx.lineWidth = 1;
  ctx.strokeRect(grid.x, grid.y, grid.size, grid.size);
  ctx.restore();
}

function drawInputAnchors() {
  const grid = getGridRect();
  const anchors = [];
  const samples = [
    [4, 4],
    [9, 7],
    [14, 6],
    [19, 8],
    [23, 5],
    [6, 14],
    [13, 14],
    [21, 15],
    [8, 22],
    [16, 23],
    [23, 21],
  ];

  samples.forEach(([col, row]) => {
    anchors.push({
      x: grid.x + col * grid.cell,
      y: grid.y - 36,
      size: 5,
    });
  });

  return anchors;
}

function render() {
  const width = VIRTUAL_WIDTH;
  const height = VIRTUAL_HEIGHT;
  if (!width || !height) {
    return;
  }

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const outputPositions = neuronPositions(10, layout.outputY, 54, 10);
  const hidden1Positions = neuronPositions(25, layout.hidden1Y, 20, 5);
  const hidden2Positions = neuronPositions(25, layout.hidden2Y, 20, 5);
  const inputAnchors = drawInputAnchors();

  drawConnections(outputPositions, hidden1Positions, "#d946ef", 1);
  drawConnections(hidden1Positions, hidden2Positions, "#22c55e", 3);
  drawConnections(hidden2Positions, inputAnchors, "#d946ef", 5);

  drawOutputSquares(outputPositions);
  drawHiddenSquares(hidden1Positions, hidden1);
  drawHiddenSquares(hidden2Positions, hidden2);
  drawInputGrid();
}

canvas.addEventListener("pointerdown", (event) => {
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  paintAt(getCanvasPoint(event));
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing) {
    return;
  }
  paintAt(getCanvasPoint(event));
});

canvas.addEventListener("pointerup", () => {
  drawing = false;
});

canvas.addEventListener("pointercancel", () => {
  drawing = false;
});

clearButton.addEventListener("click", clearDrawing);
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
loadModel();
