
const DATA = __DATA__;
const { P, H, ARM, ROD_Y, ROD_HALF, data: RUN } = DATA;
const META = RUN.meta;
const COUNT = RUN.episodes;
const stage = document.querySelector(".pb-stage");
const svg = stage.querySelector("svg");
const part = (name) => svg.querySelector(`[data-part="${name}"]`);
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const follow = (value, target, dt, seconds) => (still.matches ? target : value + (target - value) * (1 - Math.exp(-dt / seconds)));
const cam = cameraOf(P);
const VIEW = [cam.cosA * cam.cosE, cam.sinA * cam.cosE, cam.sinE];

function decode(text, Type) {
  const bytes = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
  return new Type(bytes.buffer);
}
// two bytes per episode: its length in steps, and whether it fell in the top bit
const PACKED = decode(RUN.packed, Uint16Array);
const STEPS = PACKED.map((value) => value & 0x7fff);
const FELL = PACKED.map((value) => value >> 15);
const NUMBERS = RUN.numbers ? decode(RUN.numbers, Int32Array) : null;
const numberOf = (index) => (NUMBERS ? NUMBERS[index] : RUN.first + index);
const LONGEST = STEPS.reduce((best, steps, index) => (steps > STEPS[best] ? index : best), 0);

// the drum's paper: each column the min, mean and max of a run of neighbouring episodes
const COLUMNS = (() => {
  const count = Math.min(COUNT, 1200);
  const min = new Float32Array(count);
  const mean = new Float32Array(count);
  const max = new Float32Array(count);
  for (let c = 0; c < count; c++) {
    const from = Math.floor((c * COUNT) / count);
    const to = Math.max(from + 1, Math.floor(((c + 1) * COUNT) / count));
    let low = Infinity;
    let high = -Infinity;
    let sum = 0;
    for (let index = from; index < to; index++) {
      low = Math.min(low, STEPS[index]);
      high = Math.max(high, STEPS[index]);
      sum += STEPS[index];
    }
    min[c] = low;
    max[c] = high;
    mean[c] = sum / (to - from);
  }
  return { count, min, mean, max };
})();

// ---- the pendulum -------------------------------------------------------
const rodEl = part("rod");
const bobEl = part("bob");
const carriageEl = part("carriage");
const carriageFrontEl = part("carriage-front");

// the kit's light on vertical faces: (normal . (-sinA, cosA) + 1) / 2, in four steps
const shadeOf = (nx, ny) => Math.min(3, Math.max(0, Math.floor(((nx * -cam.sinA + ny * cam.cosA + 1) / 2) * 4)));
const flat = (points) => points.map((point) => iso(point, P));

function drawRod(pivotX, angle) {
  const d = [Math.sin(angle), 0, -Math.cos(angle)];
  const n = [Math.cos(angle), 0, Math.sin(angle)];
  const [w, t] = ROD_HALF;
  const corner = (s, a, b) => [pivotX + s * ARM * d[0] + a * w * n[0], ROD_Y + b * t, H + s * ARM * d[2] + a * w * n[2]];
  const side = n[0] * VIEW[0] + n[2] * VIEW[2] > 0 ? 1 : -1;
  const front = flat([corner(0, 1, 1), corner(1, 1, 1), corner(1, -1, 1), corner(0, -1, 1)]);
  const flank = flat([corner(0, side, 1), corner(1, side, 1), corner(1, side, -1), corner(0, side, -1)]);
  const hull = pathOf(convexHull([...front, ...flank]), true);
  rodEl.innerHTML =
    `<path class="iso-fill" d="${hull}"/>` +
    `<path class="iso-shade" data-shade="${shadeOf(0, 1)}" d="${pathOf(front, true)}"/>` +
    `<path class="iso-shade" data-shade="${shadeOf(side * n[0], 0)}" d="${pathOf(flank, true)}"/>` +
    `<path class="iso-line iso-crease" data-tone="lo" d="${pathOf([flank[0], flank[1]])}"/>` +
    `<path class="iso-line iso-edge" data-tone="hi" d="${hull}"/>`;
  bobEl.setAttribute("transform", translateAlong(P, [pivotX + ARM * d[0], 0, H + ARM * d[2]]));
  const slide = translateAlong(P, [pivotX, 0, 0]);
  carriageEl.setAttribute("transform", slide);
  carriageFrontEl.setAttribute("transform", slide);
}

// ---- the motor dial, the knob ----------------------------------------------
const needleEl = part("motor-needle");
const knobEl = part("knob-pointer");
function drawNeedle(command) {
  const { x, y, r, top, up } = DATA.motor;
  const angle = up + command * (Math.PI / 3);
  needleEl.setAttribute("d", lineOnTop([x - Math.cos(angle) * 1.2, y - Math.sin(angle) * 1.2], [x + Math.cos(angle) * r, y + Math.sin(angle) * r], top + 0.1, P));
}
function drawKnob(share) {
  const { x, y, r, from, span, top } = DATA.knob;
  const angle = from + share * span;
  knobEl.setAttribute("d", lineOnTop([x + Math.cos(angle) * 2.4, y + Math.sin(angle) * 2.4], [x + Math.cos(angle) * (r - 2.6), y + Math.sin(angle) * (r - 2.6)], top, P));
}

// ---- the drum chart ---------------------------------------------------------
const chartEl = part("chart");
const penArmEl = part("pen-arm");
const penBlockEl = part("pen-block");
const penTopEl = part("pen-post-top");
const DRUM = DATA.drum;
const zOfSteps = (steps) => DRUM.paperZ[0] + (steps / META.max_steps) * (DRUM.paperZ[1] - DRUM.paperZ[0]);
const columnAt = (share) => clamp(Math.floor(share * COLUMNS.count), 0, COLUMNS.count - 1);
const facingOf = (angle) => Math.cos(angle - DRUM.facing);
const onDrum = (angle, z) => iso([DRUM.x + DRUM.r * Math.cos(angle), DRUM.y + DRUM.r * Math.sin(angle), z], P);

function drawChart(share) {
  // paper turns past the pen: an episode at share s sits span * (share - s) behind it
  const angleOf = (s) => DRUM.pen + DRUM.span * (share - s);
  const mean = [];
  const band = [];
  let run = null;
  let bandRun = null;
  const last = columnAt(share);
  for (let c = 0; c <= last; c++) {
    const s = c === last ? share : (c + 0.5) / COLUMNS.count;
    const angle = angleOf(s);
    if (facingOf(angle) < 0.01) {
      run = bandRun = null;
      continue;
    }
    if (!run) mean.push((run = []));
    if (!bandRun) band.push((bandRun = { low: [], high: [] }));
    run.push(onDrum(angle, zOfSteps(COLUMNS.mean[c])));
    bandRun.low.push(onDrum(angle, zOfSteps(COLUMNS.min[c])));
    bandRun.high.push(onDrum(angle, zOfSteps(COLUMNS.max[c])));
  }
  const grid = [];
  for (let tenth = 0; tenth <= 10; tenth++) {
    const s = tenth / 10;
    if (s > share) break;
    const angle = angleOf(s);
    if (facingOf(angle) > 0.01) grid.push(pathOf([onDrum(angle, DRUM.paperZ[0]), onDrum(angle, DRUM.paperZ[1])]));
  }
  // the paper's leading edge: the first episode
  const start = angleOf(0);
  const edge = facingOf(start) > 0.01 ? pathOf([onDrum(start, DRUM.z), onDrum(start, DRUM.z + DRUM.h)]) : "";
  chartEl.innerHTML =
    band.map(({ low, high }) => (low.length > 1 ? `<path class="pb-chart-band" d="${pathOf([...low, ...high.reverse()], true)}"/>` : "")).join("") +
    lineSvg(grid.join(""), { tone: "faint" }) +
    lineSvg(edge, { tone: "lo" }) +
    lineSvg(mean.filter((points) => points.length > 1).map((points) => pathOf(points)).join(""), { tone: "hi" });
  const steps = COLUMNS.mean[last];
  const lift = zOfSteps(steps);
  const lifted = translateAlong(P, [0, 0, lift]);
  penArmEl.setAttribute("transform", lifted);
  penBlockEl.setAttribute("transform", lifted);
  const { post, r, top, block } = DATA.pen;
  penTopEl.innerHTML = solidSvg(cylinder(post[0], post[1], r, lift + block / 2, top - lift - block / 2, P, 24), { tone: "mid", crease: "none" });
}

// ---- episodes, read from the recording when picked ------------------------------
const fmt = (value) => value.toLocaleString("en-US");
const wrap = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));
const shareOf = (index) => (COUNT > 1 ? index / (COUNT - 1) : 0);
const HOLD = 1.1; // seconds on the last frame before the replay starts again
const lastOf = (index) => `${FELL[index] ? "fell at" : "held"} ${(STEPS[index] * META.dt).toFixed(1)} s`;

// episodes live in chunk scripts next to the page, a few hundred to a file. a script
// tag loads from disk where fetch can't, so the page opens without a server; only the
// chunk holding a picked episode is loaded, and only the last few are kept
const CHUNKS = RUN.chunks;
const chunks = new Map(); // chunk number -> its episodes, oldest dropped first
const loading = new Map();
const arrived = new Map();
let offline = false;
window.__benchChunk = (number, text) => arrived.set(number, text);

function decodeChunk(number, text) {
  const bytes = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
  const from = number * CHUNKS.size;
  const to = Math.min(COUNT, from + CHUNKS.size);
  let total = 0;
  for (let index = from; index < to; index++) total += STEPS[index] + 1;
  const angles = new Int16Array(bytes.buffer, 0, total);
  const positions = new Uint16Array(bytes.buffer, total * 2, total);
  const commands = new Int8Array(bytes.buffer, total * 4, total);
  const episodes = [];
  for (let index = from, row = 0; index < to; index++) {
    const rows = STEPS[index] + 1;
    const angle = new Float32Array(rows);
    const position = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      const raw = angles[row + i] / CHUNKS.angleScale;
      // stored wrapped to ±π: unwrap, so a spinning arm turns on through the seam
      angle[i] = i ? angle[i - 1] + wrap(raw - angle[i - 1]) : raw;
      position[i] = positions[row + i] / CHUNKS.positionScale;
    }
    episodes.push({ index, episode: numberOf(index), steps: STEPS[index], fell: Boolean(FELL[index]), angle, position, command: commands.slice(row, row + rows) });
    row += rows;
  }
  return episodes;
}

function loadChunk(number) {
  if (chunks.has(number)) return Promise.resolve(chunks.get(number));
  if (loading.has(number)) return loading.get(number);
  const request = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${CHUNKS.dir}/chunk-${String(number).padStart(5, "0")}.js`;
    script.onload = () => {
      script.remove();
      const text = arrived.get(number);
      arrived.delete(number);
      if (text === undefined) return reject(new Error(`chunk ${number} did not register`));
      const episodes = decodeChunk(number, text);
      chunks.set(number, episodes);
      if (chunks.size > 8) chunks.delete(chunks.keys().next().value);
      resolve(episodes);
    };
    script.onerror = () => {
      script.remove();
      reject(new Error(`chunk ${number} is missing`));
    };
    document.head.append(script);
  })
    .catch((error) => {
      offline = true;
      throw error;
    })
    .finally(() => loading.delete(number));
  loading.set(number, request);
  return request;
}

const load = (index) => loadChunk(Math.floor(index / CHUNKS.size)).then((episodes) => episodes[index % CHUNKS.size]);

function stateOf(episode, time) {
  if (!episode) return { angle: Math.PI, position: META.rail_length / 2, command: 0 };
  const last = episode.angle.length - 1;
  if (last < 1) return { angle: episode.angle[0], position: episode.position[0], command: 0 };
  const exact = clamp(time / META.dt, 0, last);
  const step = Math.min(last - 1, Math.floor(exact));
  const t = exact - step;
  return {
    angle: episode.angle[step] + (episode.angle[step + 1] - episode.angle[step]) * t,
    position: episode.position[step] + (episode.position[step + 1] - episode.position[step]) * t,
    command: exact >= last ? 0 : episode.command[Math.floor(exact)],
  };
}

// the tour visits episodes spread across the run, the longest one among them
const TOUR = [...new Set([0, 0.04, 0.12, 0.3, 0.55, 0.8, 1].map((at) => Math.round(at * (COUNT - 1))).concat(LONGEST))].sort((a, b) => a - b);

let wanted = TOUR[0]; // the episode chosen; the pendulum switches once it has loaded
let sample = null; // the episode replaying
let share = shareOf(wanted);
let target = share;
let time = 0;
let offset = { angle: 0, position: 0, age: 1 };
let shown = stateOf(null, 0);
let command = 0;
let touring = !still.matches;
let tourStop = 0;
let steering = false; // the figure is being dragged: follow the knob once it rests
let pending = -1;
let pendingSince = 0;
let frame = 0;
let last = 0;
let visible = false;
let clock = 0;

function switchTo(next) {
  if (next === sample) return;
  const start = stateOf(next, 0);
  offset = { angle: wrap(shown.angle - start.angle), position: shown.position - start.position, age: 0 };
  sample = next;
  time = still.matches ? (next.steps * META.dt) / 2 : 0;
  run();
}

function pick(index, { keepView = false } = {}) {
  index = clamp(Math.round(index), 0, COUNT - 1);
  wanted = index;
  target = shareOf(index);
  const text = `Episode ${fmt(numberOf(index))} of ${fmt(COUNT)}: ${lastOf(index)}`;
  for (const element of [stage, plot]) {
    element.setAttribute("aria-valuenow", String(numberOf(index)));
    element.setAttribute("aria-valuetext", text);
  }
  if (!keepView) reveal(index);
  load(index).then(
    (episode) => wanted === index && switchTo(episode),
    () => run(),
  );
  placeNow();
  run();
}

function say() {
  let text;
  if (offline && (!sample || sample.index !== wanted)) text = `episode ${fmt(numberOf(wanted))} · ${lastOf(wanted)} · episode files missing: rerun uv run pendulum bench`;
  else if (!sample || sample.index !== wanted) text = `episode ${fmt(numberOf(wanted))} of ${fmt(COUNT)} · loading`;
  else {
    const length = sample.steps * META.dt;
    const seconds = Math.min(time, length).toFixed(1);
    const lean = Math.round((wrap(shown.angle - Math.PI) * 180) / Math.PI);
    const tail = time >= length ? `${sample.fell ? "fell at" : "held"} ${seconds} s` : `${seconds} s · lean ${lean > 0 ? "+" : ""}${lean}°`;
    text = `episode ${fmt(sample.episode)} of ${fmt(COUNT)} · ${tail}`;
  }
  if (readout.textContent !== text) readout.textContent = text;
}

function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  clock += dt;
  const calm = still.matches;
  const length = sample ? sample.steps * META.dt : 0;

  if (touring) {
    const stop = TOUR[tourStop];
    if (wanted !== stop) pick(stop, { keepView: true });
    if (sample && sample.index === stop && time > length + HOLD) tourStop = (tourStop + 1) % TOUR.length;
  } else if (steering) {
    // follow the knob, but only once it has rested on an episode for a moment
    const near = Math.round(share * (COUNT - 1));
    if (near !== pending) {
      pending = near;
      pendingSince = clock;
    }
    if ((calm || clock - pendingSince > 0.18) && pending !== wanted) {
      const keep = target;
      pick(pending);
      target = keep;
    }
  }

  share = follow(share, target, dt, touring ? 0.5 : 0.18);
  if (Math.abs(share - target) < 1e-6) share = target;
  if (!calm && sample) {
    time += dt;
    if (!touring && time > length + HOLD) time = 0;
    offset.age += dt;
  }

  const state = stateOf(sample, time);
  const fade = calm ? 0 : Math.exp(-offset.age / 0.16);
  shown = { angle: state.angle + offset.angle * fade, position: state.position + offset.position * fade };
  command = follow(command, state.command, dt, 0.05);

  drawRod(shown.position * DATA.M, shown.angle);
  drawNeedle(command);
  drawKnob(share);
  drawChart(share);
  say();

  const resting = calm && share === target;
  frame = visible && !resting ? requestAnimationFrame(tick) : 0;
}

function run() {
  if (!frame && visible) {
    last = 0;
    frame = requestAnimationFrame(tick);
  }
}
// the tour only runs until someone touches the bench: after that it is a picker
function hold() {
  touring = false;
  run();
}

new IntersectionObserver((entries) => {
  visible = entries[entries.length - 1].isIntersecting;
  if (visible) run();
}, { rootMargin: "120px 0px" }).observe(stage);

let drag = null;
stage.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, from: target, width: svg.getBoundingClientRect().width };
  stage.setPointerCapture(event.pointerId);
  hold();
});
stage.addEventListener("pointermove", (event) => {
  if (!drag) return;
  steering = true;
  target = clamp(drag.from + ((event.clientX - drag.x) / drag.width) * 1.25, 0, 1);
  hold();
});
const release = (event) => {
  drag = null;
  if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
};
stage.addEventListener("pointerup", release);
stage.addEventListener("pointercancel", release);

// ---- the timeline under the card ---------------------------------------------
// view is a span of episode indices; episode i covers [i, i + 1)
const plot = document.querySelector(".pb-plot");
const plotCanvas = plot.querySelector("canvas");
const nowMark = plot.querySelector('[data-kind="now"]');
const hoverMark = plot.querySelector('[data-kind="hover"]');
const tip = plot.querySelector(".pb-tip");
const axisEl = document.querySelector("[data-axis]");
const viewEl = document.querySelector("[data-view]");
const overview = document.querySelector(".pb-overview");
const overviewCanvas = overview.querySelector("canvas");
const windowEl = overview.querySelector(".pb-window");
const MIN_SPAN = Math.min(COUNT, 20);
const PLOT_TOP = 22; // room for the tip
const PLOT_FOOT = 3;
let view = { from: 0, to: COUNT };

const css = getComputedStyle(plot);
const ink = (name) => css.getPropertyValue(name).trim();
const spanOf = () => view.to - view.from;
const indexAt = (clientX) => {
  const box = plot.getBoundingClientRect();
  return clamp(Math.floor(view.from + ((clientX - box.left) / box.width) * spanOf()), 0, COUNT - 1);
};

function sizeCanvas(element, canvas) {
  const { width, height } = element.getBoundingClientRect();
  const ratio = Math.min(2, devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const g = canvas.getContext("2d");
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { g, width, height };
}

function paintPlot() {
  const { g, width, height } = sizeCanvas(plot, plotCanvas);
  const floor = height - PLOT_FOOT;
  const y = (steps) => floor - (steps / META.max_steps) * (floor - PLOT_TOP);
  const span = spanOf();
  const xOf = (index) => ((index - view.from) / span) * width;
  g.lineWidth = 1;
  g.strokeStyle = ink("--anatomy-faint");
  for (const level of [0, 0.25, 0.5, 0.75, 1]) {
    const at = Math.round(y(level * META.max_steps)) + 0.5;
    g.beginPath();
    g.moveTo(0, at);
    g.lineTo(width, at);
    g.stroke();
  }
  const first = Math.max(0, Math.floor(view.from));
  const end = Math.min(COUNT, Math.ceil(view.to));
  if (span / width <= 0.34) {
    // close enough to see each episode as a bar: held ones bright, fallen ones dim
    for (let index = first; index < end; index++) {
      const x0 = xOf(index);
      const x1 = xOf(index + 1);
      const gap = x1 - x0 > 5 ? 1 : 0;
      g.fillStyle = ink(FELL[index] ? "--anatomy-mid" : "--anatomy-hi");
      g.globalAlpha = FELL[index] ? 0.7 : 1;
      g.fillRect(x0 + gap, y(STEPS[index]), Math.max(1, x1 - x0 - 2 * gap), floor - y(STEPS[index]));
    }
    g.globalAlpha = 1;
  } else {
    // one column per pixel: the spread of its episodes, and their mean
    const columns = Math.ceil(width);
    const low = new Float32Array(columns);
    const high = new Float32Array(columns);
    const mean = new Float32Array(columns);
    for (let px = 0; px < columns; px++) {
      const from = Math.min(COUNT - 1, Math.floor(view.from + (px / width) * span));
      const to = Math.min(COUNT, Math.max(from + 1, Math.floor(view.from + ((px + 1) / width) * span)));
      let a = Infinity;
      let b = -Infinity;
      let sum = 0;
      for (let index = from; index < to; index++) {
        a = Math.min(a, STEPS[index]);
        b = Math.max(b, STEPS[index]);
        sum += STEPS[index];
      }
      low[px] = a;
      high[px] = b;
      mean[px] = sum / (to - from);
    }
    g.fillStyle = ink("--anatomy-lo");
    g.globalAlpha = 0.6;
    g.beginPath();
    for (let px = 0; px < columns; px++) g.lineTo(px + 0.5, y(high[px]));
    for (let px = columns - 1; px >= 0; px--) g.lineTo(px + 0.5, y(low[px]));
    g.fill();
    g.globalAlpha = 1;
    g.strokeStyle = ink("--anatomy-hi");
    g.beginPath();
    for (let px = 0; px < columns; px++) g.lineTo(px + 0.5, y(mean[px]));
    g.stroke();
  }
}

function paintOverview() {
  const { g, width, height } = sizeCanvas(overview, overviewCanvas);
  g.strokeStyle = ink("--anatomy-mid");
  g.lineWidth = 1;
  g.beginPath();
  for (let c = 0; c < COLUMNS.count; c++) g.lineTo(((c + 0.5) / COLUMNS.count) * width, height - 2 - (COLUMNS.mean[c] / META.max_steps) * (height - 4));
  g.stroke();
}

// axis labels on round episode numbers, about one every fifth of the view
function paintAxis() {
  const span = spanOf();
  const raw = span / 5;
  const power = 10 ** Math.floor(Math.log10(Math.max(raw, 1)));
  const step = Math.max(1, [1, 2, 5, 10].map((m) => m * power).find((value) => value >= raw));
  const labels = [`<span data-edge="start" style="left:0">${fmt(numberOf(Math.floor(view.from)))}</span>`];
  // round episode numbers when they count up from the first, round indices otherwise
  const base = NUMBERS ? 0 : RUN.first;
  for (let n = Math.ceil((view.from + base) / step) * step; n - base < view.to; n += step) {
    const at = (n - base + 0.5 - view.from) / span;
    if (at > 0.08 && at < 0.92) labels.push(`<span style="left:${(at * 100).toFixed(2)}%">${fmt(NUMBERS ? numberOf(n) : n)}</span>`);
  }
  labels.push(`<span data-edge="end" style="left:100%">${fmt(numberOf(Math.min(COUNT - 1, Math.ceil(view.to) - 1)))}</span>`);
  axisEl.innerHTML = labels.join("");
  const zoom = COUNT / span;
  viewEl.textContent = zoom <= 1.0001 ? "whole run" : `${fmt(Math.ceil(span))} shown · ${zoom.toFixed(zoom < 10 ? 1 : 0)}×`;
  windowEl.style.left = `${(view.from / COUNT) * 100}%`;
  windowEl.style.width = `${(span / COUNT) * 100}%`;
}

function placeNow() {
  const at = (wanted + 0.5 - view.from) / spanOf();
  nowMark.style.display = at < 0 || at > 1 ? "none" : "";
  nowMark.style.left = `${at * 100}%`;
}

let painted = 0;
function setView(from, to) {
  const span = clamp(to - from, MIN_SPAN, COUNT);
  const start = clamp(from, 0, COUNT - span);
  view = { from: start, to: start + span };
  // one repaint per frame however fast the wheel turns
  if (!painted)
    painted = requestAnimationFrame(() => {
      painted = 0;
      paintPlot();
      paintAxis();
      placeNow();
    });
}
const zoomAt = (factor, at) => {
  const span = clamp(spanOf() * factor, MIN_SPAN, COUNT);
  const share = (at - view.from) / spanOf();
  setView(at - share * span, at - share * span + span);
};
// keep a picked episode in view, panning as little as needed
function reveal(index) {
  const span = spanOf();
  if (index < view.from) setView(index - span * 0.1, index - span * 0.1 + span);
  else if (index + 1 > view.to) setView(index + 1 - span * 0.9, index + 1 + span * 0.1);
}

new ResizeObserver(() => {
  paintOverview();
  setView(view.from, view.to);
}).observe(plot);

function hover(event) {
  const index = indexAt(event.clientX);
  const width = plot.getBoundingClientRect().width;
  const at = ((index + 0.5 - view.from) / spanOf()) * width;
  plot.setAttribute("data-hover", "");
  hoverMark.style.left = `${at}px`;
  tip.hidden = false;
  tip.textContent = `ep ${fmt(numberOf(index))} · ${lastOf(index)}`;
  tip.style.left = `${clamp(at, tip.offsetWidth / 2 + 2, width - tip.offsetWidth / 2 - 2)}px`;
}
let press = null;
plot.addEventListener("pointerdown", (event) => {
  plot.setPointerCapture(event.pointerId);
  press = { x: event.clientX, from: view.from, moved: false };
  hold();
});
plot.addEventListener("pointermove", (event) => {
  hover(event);
  if (!press) return;
  const dx = event.clientX - press.x;
  if (Math.abs(dx) > 4) press.moved = true;
  if (press.moved) {
    plot.setAttribute("data-panning", "");
    const perPixel = spanOf() / plot.getBoundingClientRect().width;
    setView(press.from - dx * perPixel, press.from - dx * perPixel + spanOf());
  }
});
plot.addEventListener("pointerup", (event) => {
  if (press && !press.moved) {
    steering = false;
    pick(indexAt(event.clientX), { keepView: true });
  }
  press = null;
  plot.removeAttribute("data-panning");
  if (plot.hasPointerCapture(event.pointerId)) plot.releasePointerCapture(event.pointerId);
});
plot.addEventListener("pointerleave", () => {
  if (press) return;
  plot.removeAttribute("data-hover");
  tip.hidden = true;
});
plot.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    const box = plot.getBoundingClientRect();
    const perPixel = spanOf() / box.width;
    if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) setView(view.from + event.deltaX * perPixel, view.to + event.deltaX * perPixel);
    else zoomAt(Math.exp(event.deltaY * 0.002), view.from + ((event.clientX - box.left) / box.width) * spanOf());
    hover(event);
  },
  { passive: false },
);
plot.addEventListener("dblclick", (event) => zoomAt(0.25, indexAt(event.clientX) + 0.5));

let scrub = false;
const scrubTo = (event) => {
  const box = overview.getBoundingClientRect();
  const centre = ((event.clientX - box.left) / box.width) * COUNT;
  setView(centre - spanOf() / 2, centre + spanOf() / 2);
};
overview.addEventListener("pointerdown", (event) => {
  overview.setPointerCapture(event.pointerId);
  scrub = true;
  scrubTo(event);
});
overview.addEventListener("pointermove", (event) => scrub && scrubTo(event));
overview.addEventListener("pointerup", (event) => {
  scrub = false;
  if (overview.hasPointerCapture(event.pointerId)) overview.releasePointerCapture(event.pointerId);
});

document.querySelectorAll("[data-jump]").forEach((button) =>
  button.addEventListener("click", () => {
    const jump = button.getAttribute("data-jump");
    steering = false;
    hold();
    if (jump === "first") pick(0);
    else if (jump === "last") pick(COUNT - 1);
    else if (jump === "longest") pick(LONGEST);
    else if (jump === "fit") setView(0, COUNT);
    else zoomAt(jump === "in" ? 0.25 : 4, wanted + 0.5);
  }),
);

// arrows step one episode, Shift ten, Page keys a hundred; + and − zoom round the episode
function keys(event) {
  const steps = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 100, PageDown: -100 };
  let to = null;
  if (event.key in steps) to = wanted + steps[event.key] * (event.shiftKey && !event.key.startsWith("Page") ? 10 : 1);
  if (event.key === "Home") to = 0;
  if (event.key === "End") to = COUNT - 1;
  if (event.key === "+" || event.key === "=") zoomAt(0.5, wanted + 0.5);
  else if (event.key === "-" || event.key === "_") zoomAt(2, wanted + 0.5);
  else if (event.key === "0") setView(0, COUNT);
  else if (to === null) return;
  event.preventDefault();
  steering = false;
  hold();
  if (to !== null) pick(to);
}
stage.addEventListener("keydown", keys);
plot.addEventListener("keydown", keys);

still.addEventListener("change", () => {
  if (still.matches) touring = false;
  run();
});

pick(wanted, { keepView: true });
paintOverview();
tick(performance.now());
