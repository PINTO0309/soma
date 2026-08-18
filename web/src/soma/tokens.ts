// Anatomical tokens: fixed-layout numeric encoding of an assembled person.
// Port of soma/tokens.py.

import * as C from './constants';
import type { AnatomicalToken, Box, Person } from './types';

// Slot layout: sided parents get L/R slots, unsided parents one slot, +head.
export const SLOT_LAYOUT: Array<[number, number]> = (() => {
  const layout: Array<[number, number]> = [];
  for (const cid of C.KEYPOINT_PARENT_CLASS_IDS) {
    if (C.SIDE_PARENT_TO_CHILDREN.has(cid)) {
      layout.push([cid, 0]);
      layout.push([cid, 1]);
    } else {
      layout.push([cid, -1]);
    }
  }
  return layout;
})();
export const HEAD_SLOT = SLOT_LAYOUT.length;
export const N_SLOTS = SLOT_LAYOUT.length + 1; // keypoint slots + head

const SLOT_INDEX = new Map<string, number>(SLOT_LAYOUT.map(([cid, side], i) => [`${cid}:${side}`, i]));
const ANKLE_SLOTS = SLOT_LAYOUT.map(([cid], i) => (cid === 42 ? i : -1)).filter((i) => i >= 0);

function groupExtent(person: Person): Box | null {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const kp of person.keypoints) {
    xs.push(kp.x);
    ys.push(kp.y);
  }
  if (person.headBox !== null) {
    xs.push(person.headBox[0], person.headBox[2]);
    ys.push(person.headBox[1], person.headBox[3]);
  }
  if (xs.length === 0) {
    return null;
  }
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

export function buildToken(person: Person): AnatomicalToken | null {
  const boxProxy = person.bodyBox !== null ? person.bodyBox : groupExtent(person);
  if (boxProxy === null) {
    return null;
  }
  const w = Math.max(boxProxy[2] - boxProxy[0], 1.0);
  const h = Math.max(boxProxy[3] - boxProxy[1], 1.0);

  const presence = new Array<boolean>(N_SLOTS).fill(false);
  const points = new Float32Array(N_SLOTS * 2).fill(Number.NaN);

  // best-scoring keypoint per slot; unknown-side keypoints take a free side slot
  const sorted = [...person.keypoints].sort((a, b) => b.score - a.score);
  for (const kp of sorted) {
    const cid = kp.classId;
    let cand: Array<[number, number]>;
    if (C.SIDE_PARENT_TO_CHILDREN.has(cid)) {
      cand = kp.side === 0 || kp.side === 1 ? [[cid, kp.side]] : [[cid, 0], [cid, 1]];
    } else {
      cand = [[cid, -1]];
    }
    for (const [c, s] of cand) {
      const si = SLOT_INDEX.get(`${c}:${s}`);
      if (si !== undefined && !presence[si]) {
        presence[si] = true;
        points[si * 2] = (kp.x - boxProxy[0]) / w;
        points[si * 2 + 1] = (kp.y - boxProxy[1]) / h;
        break;
      }
    }
  }
  if (person.headBox !== null) {
    const hc = person.headBox;
    presence[HEAD_SLOT] = true;
    points[HEAD_SLOT * 2] = ((hc[0] + hc[2]) / 2 - boxProxy[0]) / w;
    points[HEAD_SLOT * 2 + 1] = ((hc[1] + hc[3]) / 2 - boxProxy[1]) / h;
  }

  return {
    anchor: person.anchor,
    bodyBox: person.bodyBox === null ? null : ([...person.bodyBox] as Box),
    bodyScore: person.bodyScore,
    boxProxy: [...boxProxy] as Box,
    presence,
    points,
    generation: person.generation,
    gender: person.gender,
    embedding: null, // SOMA-R: injected by the perception layer
    headBox: person.headBox === null ? null : ([...person.headBox] as Box),
    headDir: person.headDir === null ? null : [...person.headDir],
    headDirConf: person.headDirConf,
    crowding: 0,
  };
}

// Amodal full-body synthesis (v3): reconstruct the standing extent from
// anatomy (full height ~= alpha * head height); only ever EXTENDS the box
// downward. Port of soma/tokens.py apply_amodal.
export function applyAmodal(tok: AnatomicalToken, alpha: number, gamma = 0.9, overshoot = 1.0): void {
  if (tok.bodyBox === null || tok.headBox === null) {
    return;
  }
  const headH = tok.headBox[3] - tok.headBox[1];
  if (headH <= 2.0) {
    return;
  }
  if (ANKLE_SLOTS.some((s) => tok.presence[s])) {
    return; // visible ankles: the bottom edge is real
  }
  const y1 = Math.min(tok.bodyBox[1], tok.headBox[1]);
  const hEst = alpha * headH;
  if (tok.bodyBox[3] - y1 >= gamma * hEst) {
    return; // tall enough: no truncation evidence
  }
  const newY2 = y1 + overshoot * hEst;
  tok.bodyBox = [...tok.bodyBox] as Box;
  tok.bodyBox[1] = y1;
  tok.bodyBox[3] = Math.max(tok.bodyBox[3], newY2);
  tok.boxProxy = tok.bodyBox;
}
