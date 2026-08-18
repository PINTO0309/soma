// Person assembly: structured 49-class detections -> per-person part groups.
// Port of soma/assembly.py for the mask-less web detector path (wb28-family
// raw YOLO heads emit no instance masks, so keypoint->instance assignment
// uses the box-containment fallback only — identical to the python behavior
// when Detections.mask_probs is None).

import * as C from './constants';
import type { AssemblyResult, Box, Detections, Keypoint, Person } from './types';
import { detBox, detCenter } from './types';

// vocab id used in BONE_EDGE_PAIRS -> [parent keypoint class, side or null]
const VOCAB_TO_PARENT_SIDE = new Map<number, [number, number | null]>();
for (const pid of [21, 25, 35, 39, 42]) {
  VOCAB_TO_PARENT_SIDE.set(pid, [pid, null]);
}
for (const [parent, [l, r]] of C.SIDE_PARENT_TO_CHILDREN) {
  VOCAB_TO_PARENT_SIDE.set(l, [parent, 0]);
  VOCAB_TO_PARENT_SIDE.set(r, [parent, 1]);
}

function nmsClass(boxes: Box[], scores: number[], iouThr: number): number[] {
  const order = scores.map((_s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const n = order.length;
  const areas = order.map((i) => (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]));
  const alive = new Array<boolean>(n).fill(true);
  const keep: number[] = [];
  for (let k = 0; k < n; k += 1) {
    if (!alive[k]) {
      continue;
    }
    keep.push(order[k]);
    const bk = boxes[order[k]];
    for (let m = k + 1; m < n; m += 1) {
      if (!alive[m]) {
        continue;
      }
      const bm = boxes[order[m]];
      const x1 = Math.max(bk[0], bm[0]);
      const y1 = Math.max(bk[1], bm[1]);
      const x2 = Math.min(bk[2], bm[2]);
      const y2 = Math.min(bk[3], bm[3]);
      const inter = Math.max(x2 - x1, 0) * Math.max(y2 - y1, 0);
      const iou = inter / (areas[k] + areas[m] - inter + 1e-9);
      if (iou >= iouThr) {
        alive[m] = false;
      }
    }
  }
  return keep;
}

// Index of the best candidate sharing coordinates with each query, else -1.
// Gate: center distance; pick: argmax(score + IoU) among IoU > 0 candidates.
function batchSharedBoxMatch(
  det: Detections,
  qRows: number[],
  candRows: number[],
  maxDist: number,
): number[] {
  const out = new Array<number>(qRows.length).fill(-1);
  if (qRows.length === 0 || candRows.length === 0) {
    return out;
  }
  for (let qi = 0; qi < qRows.length; qi += 1) {
    const qb = detBox(det, qRows[qi]);
    const [qcx, qcy] = detCenter(det, qRows[qi]);
    const qa = (qb[2] - qb[0]) * (qb[3] - qb[1]);
    let bestVal = Number.NEGATIVE_INFINITY;
    let bestJ = -1;
    for (let cj = 0; cj < candRows.length; cj += 1) {
      const [ccx, ccy] = detCenter(det, candRows[cj]);
      if (Math.hypot(qcx - ccx, qcy - ccy) > maxDist) {
        continue;
      }
      const cb = detBox(det, candRows[cj]);
      const x1 = Math.max(qb[0], cb[0]);
      const y1 = Math.max(qb[1], cb[1]);
      const x2 = Math.min(qb[2], cb[2]);
      const y2 = Math.min(qb[3], cb[3]);
      const inter = Math.max(x2 - x1, 0) * Math.max(y2 - y1, 0);
      const ca = (cb[2] - cb[0]) * (cb[3] - cb[1]);
      const iou = inter / (qa + ca - inter + 1e-9);
      if (iou <= 0) {
        continue;
      }
      const val = det.scores[candRows[cj]] + iou;
      if (val > bestVal) {
        bestVal = val;
        bestJ = cj;
      }
    }
    out[qi] = bestJ;
  }
  return out;
}

class UnionFind {
  parent: number[];

  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_v, i) => i);
  }

  find(i: number): number {
    while (this.parent[i] !== i) {
      this.parent[i] = this.parent[this.parent[i]];
      i = this.parent[i];
    }
    return i;
  }

  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) {
      this.parent[rb] = ra;
    }
  }
}

export interface AssembleOptions {
  scoreThreshold?: number;
  attrScoreThreshold?: number;
  nmsIou?: number;
  // BYTE-style low-score pool for body anchors only.
  bodyScoreFloor?: number | null;
}

export function assemble(det: Detections, opts: AssembleOptions = {}): AssemblyResult {
  const scoreThreshold = opts.scoreThreshold ?? C.DEFAULT_SCORE_THRESHOLD;
  const attrScoreThreshold = opts.attrScoreThreshold ?? C.DEFAULT_ATTR_SCORE_THRESHOLD;
  const nmsIou = opts.nmsIou ?? C.DEFAULT_NMS_IOU;
  const bodyScoreFloor = opts.bodyScoreFloor ?? null;
  const n = det.labels.length;

  const keptRows = (classIds: Iterable<number>, thr: number): number[] => {
    const ids = new Set(classIds);
    const rows: number[] = [];
    for (let r = 0; r < n; r += 1) {
      if (ids.has(det.labels[r]) && det.scores[r] >= thr) {
        rows.push(r);
      }
    }
    return rows;
  };

  const nmsRows = (classId: number, thr: number): number[] => {
    const rows = keptRows([classId], thr);
    if (rows.length === 0) {
      return rows;
    }
    const keep = nmsClass(
      rows.map((r) => detBox(det, r)),
      rows.map((r) => det.scores[r]),
      nmsIou,
    );
    return keep.map((k) => rows[k]);
  };

  // ---- anchors (body variants), heads --------------------------------
  const bodyThr = bodyScoreFloor ?? scoreThreshold;
  let bodyRows: number[] = [];
  for (const c of C.BODY_ANCHOR_CLASS_IDS) {
    bodyRows = bodyRows.concat(nmsRows(c, bodyThr));
  }
  if (bodyRows.length > 1) {
    const keep = nmsClass(
      bodyRows.map((r) => detBox(det, r)),
      bodyRows.map((r) => det.scores[r]),
      nmsIou,
    );
    bodyRows = keep.map((k) => bodyRows[k]);
  }
  const headRows = nmsRows(C.HEAD, scoreThreshold);

  // ---- attribute lookup tables ---------------------------------------
  const attrTable = (attrs: Map<number, number>): [number[], number[]] => {
    const rows = keptRows(attrs.keys(), attrScoreThreshold);
    const vals = rows.map((r) => attrs.get(det.labels[r]) as number);
    return [rows, vals];
  };
  const [genRows, genVals] = attrTable(C.GENERATION_ATTRS);
  const [genderRows, genderVals] = attrTable(C.GENDER_ATTRS);
  const [poseRows, poseVals] = attrTable(C.HEAD_POSE_ATTRS);

  const persons: Person[] = bodyRows.map((r) => {
    const label = det.labels[r];
    const anchor = label === C.BODY ? 'body' : label === C.BODY_WHEELCHAIR ? 'wheelchair' : 'crutches';
    return {
      anchor,
      bodyRow: r,
      bodyBox: detBox(det, r),
      bodyScore: det.scores[r],
      generation: -1,
      gender: -1,
      headPose: -1,
      headRow: -1,
      headBox: null,
      headDir: null,
      headDirConf: 0,
      keypoints: [],
      boneEdges: [],
    };
  });
  const bodyPi = persons.map((_p, i) => i).filter((i) => persons[i].anchor === 'body');
  if (bodyPi.length > 0) {
    const qRows = bodyPi.map((i) => persons[i].bodyRow);
    for (const [rows, vals, field] of [
      [genRows, genVals, 'generation'],
      [genderRows, genderVals, 'gender'],
    ] as Array<[number[], number[], 'generation' | 'gender']>) {
      const match = batchSharedBoxMatch(det, qRows, rows, C.ATTR_MERGE_CENTER_DIST);
      bodyPi.forEach((pi, k) => {
        const j = match[k];
        if (j >= 0) {
          persons[pi][field] = vals[j];
        }
      });
    }
  }

  // ---- keypoints (parent classes, with L/R side from child rows) --------
  const keypoints: Keypoint[] = [];
  for (const classId of C.KEYPOINT_PARENT_CLASS_IDS) {
    const rowsP = nmsRows(classId, scoreThreshold);
    if (rowsP.length === 0) {
      continue;
    }
    const sides = new Array<number>(rowsP.length).fill(-1);
    const children = C.SIDE_PARENT_TO_CHILDREN.get(classId);
    if (children) {
      const rowsL = keptRows([children[0]], attrScoreThreshold);
      const rowsR = keptRows([children[1]], attrScoreThreshold);
      const jl = batchSharedBoxMatch(det, rowsP, rowsL, C.ATTR_MERGE_CENTER_DIST);
      const jr = batchSharedBoxMatch(det, rowsP, rowsR, C.ATTR_MERGE_CENTER_DIST);
      for (let k = 0; k < rowsP.length; k += 1) {
        if (jl[k] >= 0 && jr[k] >= 0) {
          sides[k] = det.scores[rowsL[jl[k]]] >= det.scores[rowsR[jr[k]]] ? 0 : 1;
        } else if (jl[k] >= 0) {
          sides[k] = 0;
        } else if (jr[k] >= 0) {
          sides[k] = 1;
        }
      }
    }
    rowsP.forEach((r, k) => {
      const [cx, cy] = detCenter(det, r);
      keypoints.push({
        classId,
        side: sides[k],
        x: cx,
        y: cy,
        score: det.scores[r],
        row: r,
        person: -1,
      });
    });
  }

  // ---- instance assignment: box containment (mask-less path) -----------
  const queryPts: Array<[number, number]> = keypoints
    .map((kp) => [kp.x, kp.y] as [number, number])
    .concat(headRows.map((r) => detCenter(det, r)));
  const assignPi = new Array<number>(queryPts.length).fill(-1);
  if (queryPts.length > 0 && persons.length > 0) {
    for (let qi = 0; qi < queryPts.length; qi += 1) {
      const [qx, qy] = queryPts[qi];
      let insideCount = 0;
      let owner = -1;
      for (let pi = 0; pi < persons.length; pi += 1) {
        const b = persons[pi].bodyBox;
        if (b === null) {
          continue;
        }
        if (qx >= b[0] && qx <= b[2] && qy >= b[1] && qy <= b[3]) {
          insideCount += 1;
          owner = pi;
        }
      }
      if (insideCount === 1) {
        assignPi[qi] = owner;
      }
    }
  }
  keypoints.forEach((kp, qi) => {
    kp.person = assignPi[qi];
  });

  // ---- heads -----------------------------------------------------------
  const binAngles = Array.from({ length: 8 }, (_v, i) => (i * 45.0 * Math.PI) / 180.0);
  const poseMatch = batchSharedBoxMatch(det, headRows, poseRows, C.ATTR_MERGE_CENTER_DIST);
  headRows.forEach((r, hi) => {
    const pi = assignPi[keypoints.length + hi];
    const pose = poseMatch[hi] >= 0 ? poseVals[poseMatch[hi]] : -1;
    if (pi >= 0 && persons[pi].headRow < 0) {
      persons[pi].headRow = r;
      persons[pi].headBox = detBox(det, r);
      persons[pi].headPose = pose;
      if (poseRows.length > 0) {
        const hb = detBox(det, r);
        const [hcx, hcy] = detCenter(det, r);
        const hdiag = Math.max(Math.hypot(hb[2] - hb[0], hb[3] - hb[1]), 1.0);
        let vx = 0;
        let vy = 0;
        let wsum = 0;
        for (let k = 0; k < poseRows.length; k += 1) {
          const [pcx, pcy] = detCenter(det, poseRows[k]);
          if (Math.hypot(pcx - hcx, pcy - hcy) > 0.5 * hdiag) {
            continue;
          }
          const w = det.scores[poseRows[k]];
          const th = binAngles[det.labels[poseRows[k]] - 8];
          vx += w * Math.cos(th);
          vy += w * Math.sin(th);
          wsum += w;
        }
        const norm = Math.hypot(vx, vy);
        if (norm > 1e-6) {
          persons[pi].headDir = [vx / norm, vy / norm];
          persons[pi].headDirConf = norm / Math.max(wsum, 1e-9);
        }
      }
    }
  });

  // ---- bone edges --------------------------------------------------------
  // A bone box's diagonal joins exactly two joint centers; try each vocab
  // pair against the 4 corner orientations and keep the best per bone.
  const boneRows = nmsRows(C.BONE, scoreThreshold);
  const boneEdges: Array<[number, number]> = [];
  if (boneRows.length > 0 && keypoints.length > 0) {
    const byClass = new Map<number, number[]>();
    for (let gi = 0; gi < keypoints.length; gi += 1) {
      const cid = keypoints[gi].classId;
      const arr = byClass.get(cid);
      if (arr) {
        arr.push(gi);
      } else {
        byClass.set(cid, [gi]);
      }
    }
    const nearest = (
      cid: number,
      wantSide: number | null,
      cx: number,
      cy: number,
    ): [number, number] | null => {
      const gis = byClass.get(cid);
      if (!gis) {
        return null;
      }
      let bestD = Number.POSITIVE_INFINITY;
      let bestGi = -1;
      for (const gi of gis) {
        const kp = keypoints[gi];
        if (wantSide !== null && kp.side >= 0 && kp.side !== wantSide) {
          continue;
        }
        const d = Math.hypot(cx - kp.x, cy - kp.y);
        if (d < bestD) {
          bestD = d;
          bestGi = gi;
        }
      }
      return bestGi >= 0 ? [bestD, bestGi] : null;
    };

    const results: Array<{ total: number; a: number; b: number }> = [];
    for (const r of boneRows) {
      const bb = detBox(det, r);
      const tol = Math.max(4.0, 0.25 * Math.hypot(bb[2] - bb[0], bb[3] - bb[1]));
      const c00: [number, number] = [bb[0], bb[1]];
      const c11: [number, number] = [bb[2], bb[3]];
      const c01: [number, number] = [bb[0], bb[3]];
      const c10: [number, number] = [bb[2], bb[1]];
      const orients: Array<[[number, number], [number, number]]> = [
        [c00, c11], [c11, c00], [c01, c10], [c10, c01],
      ];
      let best = { total: Number.POSITIVE_INFINITY, a: -1, b: -1 };
      for (const [va, vb] of C.BONE_EDGE_PAIRS) {
        const psA = VOCAB_TO_PARENT_SIDE.get(va);
        const psB = VOCAB_TO_PARENT_SIDE.get(vb);
        if (!psA || !psB) {
          continue;
        }
        for (const [pa, pb] of orients) {
          const ra = nearest(psA[0], psA[1], pa[0], pa[1]);
          const rb = nearest(psB[0], psB[1], pb[0], pb[1]);
          if (!ra || !rb) {
            continue;
          }
          const total = ra[0] + rb[0];
          if (ra[0] <= tol && rb[0] <= tol && total < best.total) {
            best = { total, a: ra[1], b: rb[1] };
          }
        }
      }
      if (best.a >= 0) {
        results.push(best);
      }
    }
    results.sort((x, y) => x.total - y.total);
    const usedPairs = new Set<string>();
    for (const res of results) {
      const key = `${Math.min(res.a, res.b)}:${Math.max(res.a, res.b)}`;
      if (!usedPairs.has(key)) {
        usedPairs.add(key);
        boneEdges.push([res.a, res.b]);
      }
    }
  }

  // ---- propagate person assignment along bone edges; orphan grouping -----
  const uf = new UnionFind(keypoints.length);
  for (const [a, b] of boneEdges) {
    uf.union(a, b);
  }
  const groups = new Map<number, number[]>();
  for (let gi = 0; gi < keypoints.length; gi += 1) {
    const root = uf.find(gi);
    const arr = groups.get(root);
    if (arr) {
      arr.push(gi);
    } else {
      groups.set(root, [gi]);
    }
  }

  for (const members of groups.values()) {
    const owners = new Set<number>();
    for (const m of members) {
      if (keypoints[m].person >= 0) {
        owners.add(keypoints[m].person);
      }
    }
    if (owners.size === 1) {
      const owner = owners.values().next().value as number;
      for (const m of members) {
        if (keypoints[m].person < 0) {
          keypoints[m].person = owner;
        }
      }
    }
  }

  // orphan groups (connected, but no member belongs to any body): PARTIAL persons
  for (const members of groups.values()) {
    if (members.length < 2) {
      continue;
    }
    if (members.every((m) => keypoints[m].person < 0)) {
      persons.push({
        anchor: 'orphan',
        bodyRow: -1,
        bodyBox: null,
        bodyScore: 0,
        generation: -1,
        gender: -1,
        headPose: -1,
        headRow: -1,
        headBox: null,
        headDir: null,
        headDirConf: 0,
        keypoints: [],
        boneEdges: [],
      });
      const pi = persons.length - 1;
      for (const m of members) {
        keypoints[m].person = pi;
      }
    }
  }

  for (const kp of keypoints) {
    if (kp.person >= 0) {
      persons[kp.person].keypoints.push(kp);
    }
  }
  for (const [a, b] of boneEdges) {
    const pa = keypoints[a].person;
    const pb = keypoints[b].person;
    if (pa >= 0 && pa === pb) {
      persons[pa].boneEdges.push([a, b]);
    }
  }

  return { persons, keypoints };
}
