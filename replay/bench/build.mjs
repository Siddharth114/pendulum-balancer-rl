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

// belt drive on the beam top, motor over the right pulley
const BELT_Y = -3.5;
const PULLEY_X = [-30, RAIL + 30];
const PULLEY_R = 2.5;
const BELT_Z = [BEAM_TOP + 1, BEAM_TOP + 5];
const MOTOR = { x: PULLEY_X[1], y: BELT_Y, r: 6.5, z: BEAM_TOP + 7.5, h: 15 };
const MOTOR_TOP = MOTOR.z + MOTOR.h + 1.2;
const MOTOR_DIAL_R = MOTOR.r - 1.2;

// chart drum, pen and scrub knob on the bench front
const DRUM = { x: RAIL + 22, y: 56, r: 24, z: 13, h: 52 };
const PAPER_SPAN = (300 * Math.PI) / 180;
const PAPER_Z = [DRUM.z + 4, DRUM.z + DRUM.h - 4];
const KNOB = { x: -34, y: 60, r: 13 };
const KNOB_SPAN = (270 * Math.PI) / 180;

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
  ],
  WIDTH,
  HEIGHT,
  { pad: 26, azimuth, elevation },
);
const V = G.viewOf(P);
const FACING = Math.atan2(V[1], V[0]); // floor angle that points at the viewer
const PEN_ANGLE = FACING - (38 * Math.PI) / 180; // on the visible paper, right of centre
const KNOB_FROM = FACING + (2 * Math.PI - KNOB_SPAN) / 2; // the scale's gap faces the viewer

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

function knob() {
  // a scale ring on the bench, a collar and a knurled knob whose pointer is the episode
  const { x, y, r } = KNOB;
  const out = [];
  out.push(S(k.cylinder(x, y, r + 11, 0, 0.8, P, 72, 0.4), { tone: "lo" }));
  const from = KNOB_FROM;
  const ticks = k.radialTicks(x, y, r + 11, 30, 3, 0.8, [2, 4], P, kitAngle(from), KNOB_SPAN);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(S(k.cylinder(x, y, r + 3, 0.8, 2.6, P, 64, 0.5), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, r, 3.4, 9, P, 72, 0.6), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, r, 3.4, 10.2, 96, P, 0.3), { tone: "lo" }));
  out.push(L(k.sideArc(x, y, r, 10.2, P, 48), { tone: "faint" }));
  out.push(L(k.ring(x, y, r - 2.6, 12.4, P, 64), { tone: "faint" }));
  out.push(L(k.ring(x, y, 1.8, 12.4, P, 20), { tone: "mid" }));
  return out.join("");
}

// radialTicks measures from -y, clockwise seen from above; a floor angle is from +x toward +y
function kitAngle(floor) {
  return floor + Math.PI / 2;
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

  // motor mount: a plate on four standoffs, the motor can on it, its dial on top
  const mount = { x: MOTOR.x - 9, y: MOTOR.y - 9, w: 18, d: 18, r: 1.6 };
  const legs = k.corners(mount, 1.8);
  for (const [x, y] of byDepth(legs)) out.push(S(k.cylinder(x, y, 1.1, BEAM_TOP, MOTOR.z - 1.5 - BEAM_TOP, P, 16), { tone: "mid" }));
  out.push(S(k.slabOf(mount, MOTOR.z - 1.5, 1.5, P, 4, 0.5), { tone: "mid" }));
  out.push(D(legs.map(([x, y]) => at([x, y, MOTOR.z])), { size: 0.45 }));
  const { x, y, r, z, h } = MOTOR;
  out.push(S(k.cylinder(x, y, r + 0.8, z, 2, P, 56, 0.4), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, r, z + 2, h - 2, P, 56), { tone: "mid", crease: "none" }));
  out.push(L(k.knurl(x, y, r, z + 4, z + h - 3, 40, P, 0), { tone: "faint" }));
  out.push(L([k.sideArc(x, y, r, z + 4, P, 32), k.sideArc(x, y, r, z + h - 3, P, 32)].join(""), { tone: "lo" }));
  out.push(S(k.cylinder(x, y, r + 0.4, z + h, MOTOR_TOP - z - h, P, 56, 0.5), { tone: "hi", crease: "lo" }));
  out.push(L(k.ring(x, y, MOTOR_DIAL_R, MOTOR_TOP, P, 48), { tone: "lo" }));
  // command scale: full push left to full push right over 120 degrees
  const up = FACING + Math.PI; // the dial reads from the far side, toward the viewer
  const ticks = k.radialTicks(x, y, MOTOR_DIAL_R, 8, 4, MOTOR_TOP, [0.9, 1.8], P, kitAngle(up) - Math.PI / 3, (2 * Math.PI) / 3);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(L(k.ring(x, y, 0.9, MOTOR_TOP, P, 16), { tone: "mid" }));
  // terminal block on the motor's side
  out.push(S(k.slabOf({ x: x + r - 1, y: y - 2.5, w: 3.4, d: 5, r: 0.6 }, z + 5, 5, P, 2, 0.3), { tone: "mid" }));
  out.push(D([at([x + r + 2.4, y - 1.2, z + 10]), at([x + r + 2.4, y + 1.2, z + 10])], { size: 0.4 }));
  return out.join("");
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

function drumBack() {
  const { x, y, r, z, h } = DRUM;
  const out = [];
  const plinth = { x: x - r - 8, y: y - r - 8, w: 2 * r + 16, d: 2 * r + 16, r: 5 };
  out.push(`<path class="iso-halo" d="${k.haloOf(plinth, 0, z - 2, P)}" style="filter:blur(5px)"/>`);
  out.push(S(k.slabOf(plinth, 0, z - 2, P, 6, 0.8), { tone: "mid", inner: L(k.sideSeam(plinth, 3, P, 6), { tone: "faint" }) }));
  out.push(D(k.corners(plinth, 4).map(([px, py]) => at([px, py, z - 2])), { size: 0.45 }));
  out.push(S(k.cylinder(x, y, r + 2, z - 2, 2, P, 96, 0.5), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, r, z, h, P, 120), { tone: "hi", crease: "none" }));
  // fixed chart grid: a ring at each quarter of the step limit
  const rings = [0, 0.25, 0.5, 0.75, 1].map((share) => k.sideArc(x, y, r, PAPER_Z[0] + share * (PAPER_Z[1] - PAPER_Z[0]), P, 64));
  out.push(L(rings.join(""), { tone: "faint" }));
  return out.join("");
}

function drumTop() {
  const { x, y, r, z, h } = DRUM;
  const top = z + h;
  return [
    S(k.cylinder(x, y, r + 1.6, top, 1.6, P, 96, 0.5), { tone: "mid" }),
    L(k.ring(x, y, r - 3, top + 1.6, P, 72), { tone: "faint" }),
    D(Array.from({ length: 6 }, (_, index) => {
      const angle = (index / 6) * Math.PI * 2;
      return at([x + (r - 6) * Math.cos(angle), y + (r - 6) * Math.sin(angle), top + 1.6]);
    }), { size: 0.45 }),
    S(k.cylinder(x, y, 5, top + 1.6, 3, P, 40), { tone: "mid" }),
    S(k.cylinder(x, y, 3.6, top + 4.6, 5, P, 40), { tone: "hi", crease: "lo" }),
    L(k.knurl(x, y, 3.6, top + 4.6, top + 9.6, 30, P, 0.3), { tone: "lo" }),
  ].join("");
}

const PEN_POST = [DRUM.x + (DRUM.r + 15) * Math.cos(PEN_ANGLE), DRUM.y + (DRUM.r + 15) * Math.sin(PEN_ANGLE)];
const PEN_POST_TOP = DRUM.z + DRUM.h + 6;
const PEN_POST_R = 1.6;

function penPost() {
  const [x, y] = PEN_POST;
  return [
    S(k.cylinder(x, y, 4, 0, 2.4, P, 32, 0.4), { tone: "mid" }),
    S(k.cylinder(x, y, PEN_POST_R, 2.4, PEN_POST_TOP - 2.4, P, 24), { tone: "mid", crease: "none" }),
  ].join("");
}

function penPostCap() {
  const [x, y] = PEN_POST;
  return S(k.cylinder(x, y, 2.4, PEN_POST_TOP, 1.6, P, 24), { tone: "mid" });
}

// the pen carriage, arm and nib, built at z = 0 and lifted by the live code
const PEN_BLOCK = 4;
function penArm() {
  const [px, py] = PEN_POST;
  const u = [Math.cos(PEN_ANGLE), Math.sin(PEN_ANGLE)];
  const n = [-u[1], u[0]];
  const from = [px - 2 * u[0], py - 2 * u[1]];
  const to = [DRUM.x + (DRUM.r + 1.2) * u[0], DRUM.y + (DRUM.r + 1.2) * u[1]];
  const bar = [
    [from[0] + n[0] * 1, from[1] + n[1] * 1],
    [to[0] + n[0] * 1, to[1] + n[1] * 1],
    [to[0] - n[0] * 1, to[1] - n[1] * 1],
    [from[0] - n[0] * 1, from[1] - n[1] * 1],
  ];
  const nib = [DRUM.x + (DRUM.r + 0.5) * u[0], DRUM.y + (DRUM.r + 0.5) * u[1]];
  return [
    S(k.extrude(bar, -0.8, 1.6, P), { tone: "mid" }),
    S(k.cylinder(nib[0], nib[1], 0.8, -2, 3, P, 14), { tone: "hi", crease: "lo" }),
  ].join("");
}

function penBlock() {
  const [x, y] = PEN_POST;
  const block = { x: x - 3, y: y - 3, w: 6, d: 6, r: 1 };
  return [
    S(k.slabOf(block, -PEN_BLOCK / 2, PEN_BLOCK, P, 3, 0.4), { tone: "mid" }),
    S(k.slabOf({ x: x + 3, y: y - 1, w: 2, d: 2, r: 0.4 }, -1, 2, P, 2), { tone: "mid" }),
    S(k.cylinder(x + 5.8, y, 1.4, -1.4, 2.8, P, 16), { tone: "mid" }),
    L(k.knurl(x + 5.8, y, 1.4, -1.4, 1.4, 12, P, 0.2), { tone: "lo" }),
  ].join("");
}

// the drum's limbs are fixed on screen, so one mask fades everything drawn on the paper
const drumCentre = at([DRUM.x, DRUM.y, 0]);
const drumHalf = DRUM.r * G.scaleOf(P);
const drumMask = `<defs><linearGradient id="pb-drum-fade" gradientUnits="userSpaceOnUse" x1="${(drumCentre[0] - drumHalf).toFixed(1)}" y1="0" x2="${(drumCentre[0] + drumHalf).toFixed(1)}" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".2" stop-color="#fff"/><stop offset=".8" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient><mask id="pb-drum-mask" maskUnits="userSpaceOnUse" x="0" y="0" width="${WIDTH}" height="${HEIGHT}"><rect width="${WIDTH}" height="${HEIGHT}" fill="url(#pb-drum-fade)"/></mask></defs>`;

const restPivot = [0, 0, 0];
const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label: `A pendulum on a sliding pivot. A carriage rides a beam between two posts, driven by a belt and a motor whose dial shows the push; the arm hangs from the carriage and replays episodes recorded in run ${DATA.run}. A drum chart records how long each of the ${DATA.episodes} episodes lasted, and a knob on the bench points at the episode.`,
  body: [
    drumMask,
    base(),
    post(POSTS[0], false),
    beam(),
    post(POSTS[1], true),
    drive(),
    knob(),
    `<path data-part="knob-pointer" class="iso-line" data-tone="hi" data-free="end" d=""/>`,
    `<path data-part="motor-needle" class="iso-line" data-tone="hi" data-free="end" d="" style="stroke-width:.9"/>`,
    `<g data-part="carriage">${carriageBack()}</g>`,
    `<g data-part="rod" class="iso-solid"></g>`,
    `<g data-part="bob">${bob()}</g>`,
    `<g data-part="carriage-front">${carriageFront()}</g>`,
    drumBack(),
    `<g data-part="chart" mask="url(#pb-drum-mask)"></g>`,
    drumTop(),
    `<g data-part="pen-arm">${penArm()}</g>`,
    penPost(),
    `<g data-part="pen-block">${penBlock()}</g>`,
    `<g data-part="pen-post-top"></g>`,
    penPostCap(),
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
  knob: { ...KNOB, from: KNOB_FROM, span: KNOB_SPAN, top: 12.4 },
  motor: { x: MOTOR.x, y: MOTOR.y, r: MOTOR_DIAL_R - 1, top: MOTOR_TOP, up: FACING + Math.PI },
  drum: { ...DRUM, paperZ: PAPER_Z, span: PAPER_SPAN, pen: PEN_ANGLE, facing: FACING },
  pen: { post: PEN_POST, r: PEN_POST_R, top: PEN_POST_TOP, block: PEN_BLOCK },
  data: DATA,
};

const LIVE = readFileSync(join(HERE, "live.js"), "utf8").replace("__DATA__", JSON.stringify(LIVE_DATA));

const CSS = `
.pb-stage{position:relative;touch-action:pan-y;cursor:ew-resize;outline:none}
.pb-stage:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:6px}
.pb-chart-band{fill:var(--anatomy-hi);opacity:.07;stroke:none}
.iso[data-theme="light"] .pb-chart-band{opacity:.09}
.pb-timeline{margin-top:14px;font-family:var(--anatomy-mono);font-size:10.5px;letter-spacing:.04em;color:var(--anatomy-muted);font-variant-numeric:tabular-nums}
.pb-strip-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:6px}
.pb-strip-title{text-transform:uppercase;letter-spacing:.1em}
.pb-jumps{display:flex;gap:4px;align-items:center}
.pb-jumps i{width:10px}
.pb-view{margin-left:auto;margin-right:10px}
.pb-jumps button{font:inherit;color:var(--anatomy-text);background:none;border:0;box-shadow:0 0 0 1px var(--anatomy-lo);padding:3px 8px;cursor:pointer}
.pb-jumps button:hover,.pb-jumps button:focus-visible{box-shadow:0 0 0 1px var(--anatomy-mid);color:var(--anatomy-strong);outline:none}
.pb-plot{position:relative;height:96px;background:var(--anatomy-card);box-shadow:0 0 0 1px var(--anatomy-mid);cursor:pointer;touch-action:pan-y;outline:none}
.pb-plot:focus-visible{box-shadow:0 0 0 1px var(--anatomy-hi)}
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
.pb-overview{position:relative;height:14px;margin-top:2px;box-shadow:0 0 0 1px var(--anatomy-lo);cursor:grab;touch-action:none}
.pb-overview canvas{position:absolute;inset:0;width:100%;height:100%}
.pb-window{position:absolute;top:0;bottom:0;min-width:3px;box-shadow:inset 0 0 0 1px var(--anatomy-hi);background:color-mix(in srgb,var(--anatomy-hi) 8%,transparent);pointer-events:none}
.pb-plot[data-panning]{cursor:grabbing}
`;

const body = k.plateHtml({
  fig: "Fig 1",
  title: "Pendulum bench",
  hint: "Drag, ← →, or the timeline below",
  readout: `episode ${DATA.first}`,
  keys: [
    { mark: "lit", label: "Bob: the episode replaying now" },
    { mark: "raised", label: `Drum: the longest each of the ${DATA.episodes.toLocaleString("en-US")} episodes stayed upright` },
  ],
  caption: `Run ${DATA.run}. The arm replays a recorded episode at true speed on a ${META.rail_length} m rail with a ${META.arm_length} m arm; the motor dial shows the push each step. The drum and the timeline record each episode's longest run with the bob above the rail, scaled to the run's best, out of a ${META.max_steps}-step limit; a picked episode loads from the episode files beside this page.`,
  body: `<div class="pb-stage" tabindex="0" role="slider" aria-label="Episode: drag or use the arrow keys to pick an episode, Home and End for the first and last" aria-valuemin="${DATA.first}" aria-valuemax="${DATA.last}" aria-valuenow="${DATA.first}">${svg}</div>`,
});

// the timeline: every episode's survival under the plate, zoomable down to single episodes
const fmt = (value) => value.toLocaleString("en-US");
const button = (name, label, title) => `<button type="button" data-jump="${name}" title="${title}">${label}</button>`;
const timeline = `<div class="pb-timeline">
<div class="pb-strip-head"><span class="pb-strip-title">Episodes · ${fmt(DATA.episodes)}</span><span class="pb-view" data-view></span><span class="pb-jumps">${button("first", "first", "First episode")}${button("longest", "longest", "Longest episode")}${button("last", "last", "Last episode")}<i></i>${button("out", "−", "Zoom out")}${button("in", "+", "Zoom in")}${button("fit", "fit", "Show the whole run")}</span></div>
<div class="pb-plot" tabindex="0" role="slider" aria-label="Episode timeline: click an episode to replay it, drag to pan, scroll to zoom; arrow keys step one episode, Shift ten, plus and minus zoom" aria-valuemin="${DATA.first}" aria-valuemax="${DATA.last}" aria-valuenow="${DATA.first}">
<canvas></canvas><span class="pb-scale" style="top:4px">${META.max_steps} steps upright</span><span class="pb-mark" data-kind="hover"></span><span class="pb-mark" data-kind="now"></span><span class="pb-tip" hidden></span>
</div>
<div class="pb-axis" data-axis></div>
<div class="pb-overview" aria-hidden="true"><canvas></canvas><span class="pb-window"></span></div>
</div>`;
const page = body.replace("<figcaption", `${timeline}<figcaption`);

writeFileSync(outPath, k.pageHtml({ title: "Pendulum bench", theme, width: 860, body: `<style>${CSS}</style>${page}`, script: readFileSync(join(HERE, "kit/iso-kit.mjs"), "utf8").replace(/^export /gm, "") + LIVE }));
console.log("wrote", outPath);
