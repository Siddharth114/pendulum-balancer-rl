// node replay/bench/build.mjs <bench.json> <out.html> [--light]
// draws the pendulum bench around one run's condensed data (replay/bench/export.py)
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./kit/iso-kit.mjs";
import * as G from "./kit/lathe.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const [dataPath, outPath] = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
const theme = process.argv.includes("--light") ? "light" : "dark";
const DATA = JSON.parse(readFileSync(dataPath, "utf8"));
const META = DATA.meta;

// world units: 30 per metre, x along the rail, y toward the front, z up
const M = 30;
const RAIL = META.rail_length * M;
const ARM = META.arm_length * M;
const BOB_R = 7;
const H = ARM + BOB_R + 27; // pivot height, so the hanging bob clears the bench
const BEAM_Z = H + 2;
const BEAM_TOP = H + 16;
const BEAM = { x: -36, y: -8, w: RAIL + 72, d: 10, r: 1.5 };
const POST_W = 16;
const POSTS = [
  { x: BEAM.x - POST_W, y: -16, w: POST_W, d: 22, r: 2 },
  { x: BEAM.x + BEAM.w, y: -16, w: POST_W, d: 22, r: 2 },
];
const BASE = { x: -ARM - 40, y: -32, w: RAIL + 2 * ARM + 80, d: 124, r: 10 };
const BASE_H = 12;
const FOOT_H = 5;

// the swing plane and what sits on the carriage, front to back
const ROD_Y = 17;
const PLATE = { x: -19, y: 4, w: 38, d: 8, r: 2.4 };
const PLATE_Z = [H - 17, H + 25];
const DIAL_Y = [12, 12.8];
const DIAL_R = 12.5;
const HOUSING = [12.8, 15];
const HUB = [15.2, 18.6];
const ROD_HALF = [1.5, 1.2]; // in the swing plane, across it
const BOB_Y = [13.5, 20.5];
const CAP = [18.6, 21];

// belt drive on the beam top, a bracket over the left pulley, the motor over the right
const BELT_Y = -3.5;
const PULLEY_X = [-30, RAIL + 30];
const PULLEY_R = 2.5;
const BELT_Z = [BEAM_TOP + 1, BEAM_TOP + 5];
const MOTOR = { x: PULLEY_X[1], y: BELT_Y, r: 7.5, z: BEAM_TOP + 7.5, h: 20 };
// the push gauge on a stand above the motor, facing the viewer like the protractor
const GAUGE_R = 18;
const GAUGE_STAND = MOTOR.z + MOTOR.h + 3; // where the stand leaves the motor's cap
const GAUGE = { x: MOTOR.x, z: GAUGE_STAND + 3 + GAUGE_R, back: MOTOR.y - 1.2, face: MOTOR.y + 1.2, r: GAUGE_R };
const GAUGE_SWING = Math.PI / 3; // full push left or right swings the needle 60 degrees

const azimuth = 62;
const elevation = 27;
const WIDTH = 760;
const HEIGHT = 470;
const P = k.fitProjection(
  [
    ...k.boxCorners(BASE, -BASE_H - FOOT_H, 0),
    [BEAM.x, 0, H + ARM + BOB_R + 4],
    [BEAM.x + BEAM.w, 0, H + ARM + BOB_R + 4],
    [-ARM - BOB_R, ROD_Y, H + ARM + BOB_R],
    [MOTOR.x + GAUGE_R, MOTOR.y, MOTOR.z + MOTOR.h + 6 + 2 * GAUGE_R + 2],
  ],
  WIDTH,
  HEIGHT,
  { pad: 26, azimuth, elevation },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const along = (o, a) => G.frameAlong(o, a);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const disc = (s0, s1, r, F, style = {}) => S(G.disc(s0, s1, r, F, P), style);
// a disc on the y axis: everything on the carriage faces the viewer along +y
const yDisc = (x, z, y0, y1, r, style) => disc(y0, y1, r, along([x, 0, z], [0, 1, 0]), style);
const yRing = (x, z, y, r) => G.circleOf(along([x, 0, z], [0, 1, 0]), y, r, P);

function base() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(BASE, 26))) {
    out.push(S(k.cylinder(x, y, 8, -BASE_H - FOOT_H, 1.6, P, 32), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 6.2, -BASE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 32), { tone: "mid" }));
  }
  out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 10, 1.6), { tone: "mid", inner: L(k.sideSeam(BASE, -BASE_H + 3.5, P, 10), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(BASE, 6), 0, P, 10), { tone: "lo" }));
  // an optical breadboard: tapped holes on a 10 unit pitch inside the groove
  const holes = k.insetPlan(BASE, 14);
  const pitch = 10;
  const columns = Math.floor(holes.w / pitch);
  const rows = Math.floor(holes.d / pitch);
  const grid = k.dotGrid(holes.x + (holes.w - columns * pitch) / 2, holes.y + (holes.d - rows * pitch) / 2, columns + 1, rows + 1, pitch, 0, P);
  out.push(D(grid, { size: 0.55, tone: "lo" }));
  const screws = k.corners(BASE, 13);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 18)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));

  // label plate: three rule lines stand for the engraving
  const label = { x: 104, y: BASE.y + BASE.d - 26, w: 74, d: 14, r: 1.4 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 4), { tone: "lo" }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.6), 0.8, P, 4), { tone: "faint" }));
  [[0, 52], [0, 38], [0, 44]].forEach(([from, to], row) => {
    const y = label.y + 4 + row * 3;
    out.push(L(k.lineOnTop([label.x + 7 + from, y], [label.x + 7 + to, y], 0.8, P), { tone: "lo", free: true }));
  });
  out.push(D([at([label.x + 3.4, label.y + label.d / 2, 0.8]), at([label.x + label.w - 3.4, label.y + label.d / 2, 0.8])], { size: 0.45 }));
  return out.join("");
}

function post(plan, right) {
  const out = [];
  const flange = k.insetPlan(plan, -4);
  out.push(S(k.slabOf(flange, 0, 3, P, 4, 0.6), { tone: "mid" }));
  out.push(D(k.corners(flange, 2.4).map(([x, y]) => at([x, y, 3])), { size: 0.45 }));
  out.push(S(k.slabOf(plan, 3, BEAM_TOP - 3, P, 4, 0.5), { tone: "mid", inner: L([k.sideSeam(plan, 26, P, 4), k.sideSeam(plan, BEAM_Z - 6, P, 4)].join(""), { tone: "faint" }) }));
  // cap plate and two bolts that hold the beam end
  out.push(S(k.slabOf(k.insetPlan(plan, -1), BEAM_TOP, 2, P, 4, 0.5), { tone: "mid" }));
  const face = plan.y + plan.d;
  out.push(D([[BEAM_Z + 4], [BEAM_TOP - 4]].map(([z]) => at([plan.x + plan.w / 2, face, z])), { size: 0.5 }));
  // a gusset on the inside, under the beam
  const gx = right ? plan.x - 6 : plan.x + plan.w;
  out.push(S(k.slabOf({ x: gx, y: -6, w: 6, d: 6, r: 0.6 }, BEAM_Z - 14, 14, P, 2, 0.3), { tone: "mid" }));
  return out.join("");
}

function beam() {
  const out = [];
  out.push(S(k.slabOf(BEAM, BEAM_Z, BEAM_TOP - BEAM_Z, P, 3, 0.6), { tone: "mid", inner: L(k.sideSeam(BEAM, BEAM_Z + 7, P, 3), { tone: "faint" }) }));
  // ruler strip on the front of the top: a tick every 0.2 m, numbered metres as long ticks
  const strip = { x: -3, y: -0.6, w: RAIL + 6, d: 2.2, r: 0.4 };
  out.push(S(k.slabOf(strip, BEAM_TOP, 0.3, P, 2), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(0, RAIL, M * 0.2, 5, strip.y, BEAM_TOP + 0.3, [1.1, 2.2], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(L(k.lineOnTop([RAIL / 2, strip.y], [RAIL / 2, strip.y + 2.2], BEAM_TOP + 0.3, P), { tone: "hi" }));
  // screws along the front face, every metre
  const screws = [];
  for (let x = 0; x <= RAIL + 1e-6; x += M) screws.push(at([x, BEAM.y + BEAM.d, BEAM_Z + 3.4]));
  out.push(D(screws, { size: 0.4, tone: "lo" }));
  // end stops with rubber bumpers, where the pivot would run out of rail
  for (const x of [-27, RAIL + 22]) {
    const stop = { x, y: 2, w: 5, d: 4, r: 0.8 };
    out.push(S(k.slabOf(stop, BEAM_Z + 2, 10, P, 2, 0.4), { tone: "mid" }));
    out.push(D([at([x + 2.5, 6, BEAM_Z + 9.5]), at([x + 2.5, 6, BEAM_Z + 4.5])], { size: 0.4 }));
    out.push(yDisc(x + 2.5, BEAM_Z + 7, 6, 8, 1.8, { tone: "lo" }));
  }
  return out.join("");
}

function drive() {
  const out = [];
  // belt: two runs on the beam top, wrapped round a pulley at each end
  for (const y of [BELT_Y - PULLEY_R - 0.4, BELT_Y + PULLEY_R]) {
    out.push(S(k.slabOf({ x: PULLEY_X[0], y, w: PULLEY_X[1] - PULLEY_X[0], d: 0.4, r: 0 }, BELT_Z[0], BELT_Z[1] - BELT_Z[0], P, 1), { tone: "lo", crease: "none" }));
  }
  for (const x of PULLEY_X) {
    out.push(S(k.cylinder(x, BELT_Y, 1.4, BEAM_TOP, 1, P, 20), { tone: "mid" }));
    out.push(S(k.cylinder(x, BELT_Y, PULLEY_R + 1.1, BELT_Z[0] - 0.6, 0.6, P, 40), { tone: "mid" }));
    out.push(S(k.cylinder(x, BELT_Y, PULLEY_R + 0.4, BELT_Z[0], BELT_Z[1] - BELT_Z[0], P, 40), { tone: "lo", crease: "none" }));
    out.push(S(k.cylinder(x, BELT_Y, PULLEY_R + 1.1, BELT_Z[1], 0.6, P, 40), { tone: "mid" }));
    out.push(S(k.cylinder(x, BELT_Y, 1, BELT_Z[1] + 0.6, 0.8, P, 16), { tone: "mid" }));
  }
  // idler bracket over the left pulley
  const left = { x: PULLEY_X[0] - 5, y: BELT_Y - 5, w: 10, d: 10, r: 1 };
  for (const [x, y] of k.corners(left, 1.4)) out.push(S(k.cylinder(x, y, 0.9, BEAM_TOP, BELT_Z[1] + 1.4 - BEAM_TOP, P, 14), { tone: "mid" }));
  out.push(S(k.slabOf(left, BELT_Z[1] + 1.4, 1.4, P, 3, 0.4), { tone: "mid" }));
  out.push(D(k.corners(left, 1.4).map(([x, y]) => at([x, y, BELT_Z[1] + 2.8])), { size: 0.4 }));

  // motor mount: a plate on four standoffs, the motor can on it, a capped top
  const mount = { x: MOTOR.x - 10, y: MOTOR.y - 10, w: 20, d: 20, r: 1.6 };
  const legs = k.corners(mount, 1.8);
  for (const [x, y] of byDepth(legs)) out.push(S(k.cylinder(x, y, 1.1, BEAM_TOP, MOTOR.z - 1.5 - BEAM_TOP, P, 16), { tone: "mid" }));
  out.push(S(k.slabOf(mount, MOTOR.z - 1.5, 1.5, P, 4, 0.5), { tone: "mid" }));
  out.push(D(legs.map(([x, y]) => at([x, y, MOTOR.z])), { size: 0.45 }));
  const { x, y, r, z, h } = MOTOR;
  out.push(S(k.cylinder(x, y, r + 0.8, z, 2, P, 56, 0.4), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, r, z + 2, h - 2, P, 56), { tone: "mid", crease: "none" }));
  out.push(L(k.knurl(x, y, r, z + 4, z + h - 3, 40, P, 0), { tone: "faint" }));
  out.push(S(k.cylinder(x, y, r + 0.4, z + h, 1.6, P, 56, 0.5), { tone: "hi", crease: "lo" }));
  out.push(S(k.cylinder(x, y, 3, z + h + 1.6, 1.4, P, 24), { tone: "mid" }));
  // terminal block on the motor's side
  out.push(S(k.slabOf({ x: x + r - 1, y: y - 2.5, w: 3.4, d: 5, r: 0.6 }, z + 5, 5, P, 2, 0.3), { tone: "mid" }));
  out.push(D([at([x + r + 2.4, y - 1.2, z + 10]), at([x + r + 2.4, y + 1.2, z + 10])], { size: 0.4 }));
  return out.join("");
}

// a point on the gauge face: angle from straight up, positive toward +x (push right)
const onGauge = (angle, radius, lift = 0) => at([GAUGE.x + radius * Math.sin(angle), GAUGE.face + 0.4 + lift, GAUGE.z + radius * Math.cos(angle)]);

function gauge() {
  const { x, z, back, face, r } = GAUGE;
  const out = [];
  out.push(S(k.cylinder(x, MOTOR.y, 1.8, GAUGE_STAND, z - GAUGE_STAND, P, 20), { tone: "mid", crease: "none" }));
  out.push(yDisc(x, z, back, face, r, { tone: "mid" }));
  out.push(yDisc(x, z, face, face + 0.4, r - 1.4, { tone: "hi", crease: "lo" }));
  // scale: a tick every 15 degrees over the swing, long at full left, zero and full right
  const minor = [];
  const major = [];
  for (let step = -4; step <= 4; step++) {
    const angle = (step / 4) * GAUGE_SWING;
    const long = step % 4 === 0;
    (long ? major : minor).push(k.pathOf([onGauge(angle, r * (long ? 0.7 : 0.78)), onGauge(angle, r * 0.88)]));
  }
  out.push(L(minor.join(""), { tone: "mid" }), L(major.join(""), { tone: "hi" }));
  // L and R past the ends of the scale
  for (const [side, label] of [[-1, "L"], [1, "R"]]) {
    const [tx, ty] = onGauge(side * (GAUGE_SWING + 0.36), r * 0.66);
    out.push(`<text class="iso-face-text pb-gauge-text" x="${tx.toFixed(1)}" y="${ty.toFixed(1)}">${label}</text>`);
  }
  return out.join("");
}

function gaugeHub() {
  return [yDisc(GAUGE.x, GAUGE.z, GAUGE.face + 0.4, GAUGE.face + 1.6, 2.6, { tone: "hi", crease: "lo" }), D([onGauge(0, 0, 1.2)], { size: 0.5, tone: "hi" })].join("");
}

function carriageBack() {
  // built at pivot x = 0, moved along the rail by the live code
  const out = [];
  const [z0, z1] = PLATE_Z;
  out.push(S(k.slabOf({ x: -12, y: -9, w: 24, d: 13, r: 1.6 }, BELT_Z[1] + 1, 3, P, 3, 0.5), { tone: "mid" }));
  out.push(D([[-8, -6], [8, -6], [-8, 1], [8, 1]].map(([x, y]) => at([x, y, BELT_Z[1] + 4])), { size: 0.4 }));
  // belt clamp under the bridge, on the near run
  out.push(S(k.slabOf({ x: -6, y: BELT_Y + PULLEY_R - 1, w: 12, d: 2.6, r: 0.4 }, BELT_Z[0] - 0.4, BELT_Z[1] - BELT_Z[0] + 1.4, P, 2), { tone: "lo" }));
  out.push(S(k.slabOf(PLATE, z0, z1 - z0, P, 3, 0.6), { tone: "mid", inner: L(k.sideSeam(PLATE, z0 + 3, P, 3), { tone: "faint" }) }));
  // roller axles through the plate, riding above and below the beam
  for (const [x, z] of [[-15, BEAM_TOP + 3], [15, BEAM_TOP + 3], [-15, BEAM_Z - 3], [15, BEAM_Z - 3]]) {
    out.push(yDisc(x, z, PLATE.y + PLATE.d, PLATE.y + PLATE.d + 1, 2.2, { tone: "mid" }));
    out.push(D([at([x, PLATE.y + PLATE.d + 1, z])], { size: 0.45, tone: "mid" }));
  }
  // protractor: upright is the zero, every 15 degrees, long every 45
  out.push(yDisc(0, H, DIAL_Y[0], DIAL_Y[1], DIAL_R, { tone: "hi", crease: "lo" }));
  const minor = [];
  const major = [];
  for (let index = 0; index < 24; index++) {
    const angle = (index / 24) * Math.PI * 2;
    const long = index % 3 === 0;
    const inner = DIAL_R - (long ? 2.6 : 1.4);
    const point = (r) => at([r * Math.sin(angle), DIAL_Y[1], H + r * Math.cos(angle)]);
    (index === 0 ? [] : long ? major : minor).push(k.pathOf([point(inner), point(DIAL_R - 0.1)]));
  }
  out.push(L(minor.join(""), { tone: "lo" }), L(major.join(""), { tone: "mid" }));
  out.push(L(k.pathOf([at([0, DIAL_Y[1], H + DIAL_R - 3.6]), at([0, DIAL_Y[1], H + DIAL_R - 0.1])]), { tone: "hi" }));
  out.push(L(yRing(0, H, DIAL_Y[1], DIAL_R - 3.2), { tone: "faint" }));
  out.push(yDisc(0, H, HOUSING[0], HOUSING[1], 6, { tone: "mid" }));
  out.push(D(G.dotsOf(along([0, 0, H], [0, 1, 0]), HOUSING[1], 4.4, 4, P, { all: true, phase: Math.PI / 4 }), { size: 0.4 }));
  return out.join("");
}

function carriageFront() {
  return [
    yDisc(0, H, HUB[0], HUB[1], 3.8, { tone: "hi", crease: "lo" }),
    yDisc(0, H, CAP[0], CAP[1], 2.4, { tone: "mid" }),
    D([at([0, CAP[1], H])], { size: 0.5, tone: "hi" }),
  ].join("");
}

function bob() {
  // built at the origin, moved to the arm's end by the live code
  return [
    `<path class="iso-halo" d="${G.disc(BOB_Y[0], BOB_Y[1], BOB_R, along([0, 0, 0], [0, 1, 0]), P).fill}" style="filter:blur(3px);opacity:.35"/>`,
    yDisc(0, 0, BOB_Y[0], BOB_Y[1], BOB_R, { tone: "hi", lit: true }),
    L(yRing(0, 0, BOB_Y[1], BOB_R - 2.2), { tone: "lo" }),
    D([at([0, BOB_Y[1], 0])], { size: 0.55, tone: "hi" }),
  ].join("");
}

const restPivot = [0, 0, 0];
const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label: `A pendulum on a sliding pivot. A carriage rides a beam between two posts, driven by a belt and a motor whose gauge shows the push left or right; the arm hangs from the carriage and replays episodes recorded in run ${DATA.run}.`,
  body: [
    base(),
    post(POSTS[0], false),
    beam(),
    post(POSTS[1], true),
    drive(),
    gauge(),
    `<path data-part="push-sector" class="pb-push" d=""/>`,
    `<path data-part="motor-needle" class="iso-line pb-needle" d=""/>`,
    gaugeHub(),
    `<g data-part="carriage">${carriageBack()}</g>`,
    `<g data-part="rod" class="iso-solid"></g>`,
    `<g data-part="bob">${bob()}</g>`,
    `<g data-part="carriage-front">${carriageFront()}</g>`,
  ],
});

const LIVE_DATA = {
  P,
  M,
  H,
  ARM,
  ROD_Y,
  ROD_HALF,
  HUB,
  restPivot,
  gauge: { x: GAUGE.x, y: GAUGE.face + 0.5, z: GAUGE.z, r: GAUGE.r, swing: GAUGE_SWING },
  data: DATA,
};

const LIVE = readFileSync(join(HERE, "live.js"), "utf8").replace("__DATA__", JSON.stringify(LIVE_DATA));

const CSS = `
.pb-stage{position:relative;touch-action:pan-y;cursor:ew-resize;outline:none}
.pb-stage:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:6px}
.pb-push{fill:var(--anatomy-blue);opacity:.55;stroke:none}
.pb-needle{stroke:var(--anatomy-lit);stroke-width:1.8}
.pb-gauge-text{font-size:8px;font-weight:600;text-anchor:middle;dominant-baseline:central;fill:var(--anatomy-strong)}
.pb-plate-hint{display:flex;align-items:center;gap:10px;color:var(--anatomy-strong)!important;font-size:11.5px!important;text-transform:none!important;letter-spacing:.02em!important}
.pb-plate-hint span{display:inline-flex;align-items:center;gap:5px}
.pb-plate-hint b{font-weight:400;color:var(--anatomy-muted)}
.pb-plate-hint kbd{font:inherit;font-size:10.5px;line-height:1;min-width:18px;text-align:center;padding:3px 5px;color:var(--anatomy-strong);box-shadow:0 0 0 1px var(--anatomy-mid),0 1.5px 0 0 var(--anatomy-mid)}
.pb-timeline{margin-top:16px;font-family:var(--anatomy-mono);font-size:10.5px;letter-spacing:.04em;color:var(--anatomy-muted);font-variant-numeric:tabular-nums}
.pb-strip-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px 16px;margin-bottom:10px}
.pb-strip-title{text-transform:uppercase;letter-spacing:.1em;color:var(--anatomy-text)}
.pb-view{margin-left:auto}
.pb-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px}
.pb-group{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
.pb-group-label{text-transform:uppercase;letter-spacing:.1em;margin-right:2px}
.pb-group button{font:inherit;font-size:12px;line-height:1;white-space:nowrap;color:var(--anatomy-strong);background:var(--anatomy-face);border:0;box-shadow:0 0 0 1px var(--anatomy-mid);padding:8px 12px;min-height:30px;cursor:pointer;transition:box-shadow .15s var(--anatomy-ease),background .15s var(--anatomy-ease)}
.pb-group button[data-wide]{min-width:30px;padding:8px 10px}
.pb-group button:hover{box-shadow:0 0 0 1px var(--anatomy-hi)}
.pb-group button:focus-visible{box-shadow:0 0 0 2px var(--anatomy-lit);outline:none}
.pb-group button:active{background:var(--anatomy-card);transform:translateY(.5px)}
.pb-usage{position:absolute;top:5px;right:8px;color:var(--anatomy-text);pointer-events:none;transition:opacity .15s}
.pb-plot[data-hover] .pb-usage{opacity:0}
@media (max-width:560px){.pb-usage{display:none}.pb-group-label{flex-basis:100%}}
.pb-plot{position:relative;height:110px;background:var(--anatomy-card);box-shadow:0 0 0 1px var(--anatomy-mid);cursor:pointer;touch-action:pan-y;outline:none}
.pb-plot:hover{box-shadow:0 0 0 1px var(--anatomy-hi)}
.pb-plot:focus-visible{box-shadow:0 0 0 2px var(--anatomy-lit)}
.pb-plot canvas{position:absolute;inset:0;width:100%;height:100%}
.pb-mark{position:absolute;top:0;bottom:0;width:0;pointer-events:none;border-left:1px solid var(--anatomy-lit)}
.pb-mark[data-kind="hover"]{border-left:1px dashed var(--anatomy-mid);display:none}
.pb-plot[data-hover] .pb-mark[data-kind="hover"]{display:block}
.pb-tip{position:absolute;top:6px;white-space:nowrap;color:var(--anatomy-strong);background:var(--anatomy-card);padding:1px 5px;box-shadow:0 0 0 1px var(--anatomy-lo);transform:translateX(-50%)}
.pb-scale{position:absolute;left:6px;color:var(--anatomy-muted);pointer-events:none}
.pb-axis{position:relative;height:16px;margin-top:4px;overflow:hidden}
.pb-axis span{position:absolute;transform:translateX(-50%)}
.pb-axis span[data-edge="start"]{transform:none}
.pb-axis span[data-edge="end"]{transform:translateX(-100%)}
.pb-overview{position:relative;height:20px;margin-top:4px;box-shadow:0 0 0 1px var(--anatomy-mid);cursor:grab;touch-action:none}
.pb-overview:active{cursor:grabbing}
.pb-overview canvas{position:absolute;inset:0;width:100%;height:100%}
.pb-window{position:absolute;top:0;bottom:0;min-width:6px;box-shadow:inset 0 0 0 1.5px var(--anatomy-lit);background:color-mix(in srgb,var(--anatomy-hi) 16%,transparent);pointer-events:none}
.pb-plot[data-panning]{cursor:grabbing}
`;

const body = k.plateHtml({
  fig: "Fig 1",
  title: "Pendulum bench",
  readout: `episode ${DATA.first}`,
  keys: [
    { mark: "lit", label: "Bob: the episode replaying now" },
    { mark: "blue", label: "Motor gauge: the push each step, left or right" },
  ],
  caption: `Run ${DATA.run}. The arm replays a recorded episode at true speed on a ${META.rail_length} m rail with a ${META.arm_length} m arm; the motor gauge swings left or right with the push each step. The timeline records each episode's longest run with the bob above the rail, scaled to the run's best, out of a ${META.max_steps}-step limit; a picked episode loads from the episode files beside this page.`,
  body: `<div class="pb-stage" tabindex="0" role="slider" aria-label="Episode: drag or use the arrow keys to pick an episode, Home and End for the first and last" aria-valuemin="${DATA.first}" aria-valuemax="${DATA.last}" aria-valuenow="${DATA.first}">${svg}</div>`,
});

// the timeline: every episode's survival under the plate, zoomable down to single episodes
const fmt = (value) => value.toLocaleString("en-US");
const button = (name, label, title, wide = false) => `<button type="button" data-jump="${name}" title="${title}" aria-label="${title}"${wide ? " data-wide" : ""}>${label}</button>`;
const timeline = `<div class="pb-timeline">
<div class="pb-strip-head"><span class="pb-strip-title">Episodes · ${fmt(DATA.episodes)}</span><span class="pb-view" data-view></span></div>
<div class="pb-strip-head"><span class="pb-controls"><span class="pb-group"><span class="pb-group-label">Episode</span>${button("first", "first", "First episode")}${button("prev", "‹ prev", "Previous episode")}${button("next", "next ›", "Next episode")}${button("last", "last", "Last episode")}${button("longest", "★ longest", "Longest episode")}</span><span class="pb-group"><span class="pb-group-label">Zoom</span>${button("out", "−", "Zoom out", true)}${button("in", "+", "Zoom in", true)}${button("fit", "fit", "Show the whole run")}</span></span></div>
<div class="pb-plot" tabindex="0" role="slider" aria-label="Episode timeline: click an episode to replay it, drag to pan, scroll to zoom; arrow keys step one episode, Shift ten, plus and minus zoom" aria-valuemin="${DATA.first}" aria-valuemax="${DATA.last}" aria-valuenow="${DATA.first}">
<canvas></canvas><span class="pb-scale" style="top:4px">${META.max_steps} steps upright</span><span class="pb-usage">click to replay · drag to pan · scroll to zoom</span><span class="pb-mark" data-kind="hover"></span><span class="pb-mark" data-kind="now"></span><span class="pb-tip" hidden></span>
</div>
<div class="pb-axis" data-axis></div>
<div class="pb-overview" aria-hidden="true"><canvas></canvas><span class="pb-window"></span></div>
</div>`;
const plateHint = `<span class="iso-plate-corner pb-plate-hint" data-corner="hint"><span><kbd>drag</kbd><b>the bench</b></span><span><kbd>←</kbd><kbd>→</kbd><b>step</b></span><span><kbd>shift</kbd><b>×10</b></span></span>`;
const page = body
  .replace('<div class="iso-plate-corners" aria-hidden="true">', `$&${plateHint}`)
  .replace("<figcaption", `${timeline}<figcaption`);

writeFileSync(outPath, k.pageHtml({ title: "Pendulum bench", theme, width: 860, body: `<style>${CSS}</style>${page}`, script: readFileSync(join(HERE, "kit/iso-kit.mjs"), "utf8").replace(/^export /gm, "") + LIVE }));
console.log("wrote", outPath);
