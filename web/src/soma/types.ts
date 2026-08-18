// Shared numeric types of the SOMA web port. Boxes are [x1, y1, x2, y2] in
// frame coordinates, matching the python package.

export type Box = [number, number, number, number];

// All thresholded rows of one frame, boxes mapped back to frame coords
// (soma/detector.py Detections, mask-less web variant).
export interface Detections {
  labels: Int32Array;
  boxes: Float32Array; // (N, 4) flattened
  scores: Float32Array;
  // frame -> detector-input mapping: p_in = p_frame * scale + offset
  scale: [number, number];
  offset: [number, number];
}

export function detBox(det: Detections, row: number): Box {
  const o = row * 4;
  return [det.boxes[o], det.boxes[o + 1], det.boxes[o + 2], det.boxes[o + 3]];
}

export function detCenter(det: Detections, row: number): [number, number] {
  const o = row * 4;
  return [(det.boxes[o] + det.boxes[o + 2]) * 0.5, (det.boxes[o + 1] + det.boxes[o + 3]) * 0.5];
}

export interface Keypoint {
  classId: number;
  side: number; // 0 left, 1 right, -1 unknown
  x: number;
  y: number;
  score: number;
  row: number;
  person: number; // index into persons, -1 = orphan
}

export type AnchorKind = 'body' | 'wheelchair' | 'crutches' | 'orphan';

export interface Person {
  anchor: AnchorKind;
  bodyRow: number;
  bodyBox: Box | null;
  bodyScore: number;
  generation: number; // 0 adult, 1 child, -1 unknown
  gender: number;     // 0 male, 1 female, -1 unknown
  headPose: number;   // 0..7, -1 unknown
  headRow: number;
  headBox: Box | null;
  headDir: [number, number] | null; // unit (cos, sin)
  headDirConf: number;
  keypoints: Keypoint[];
  boneEdges: Array<[number, number]>;
}

export interface AssemblyResult {
  persons: Person[];
  keypoints: Keypoint[];
}

// Fixed-layout numeric encoding of an assembled person (soma/tokens.py).
export interface AnatomicalToken {
  anchor: AnchorKind;
  bodyBox: Box | null;
  bodyScore: number;
  boxProxy: Box;
  presence: boolean[];        // (N_SLOTS,)
  points: Float32Array;       // (N_SLOTS, 2) normalized to boxProxy, NaN if absent
  generation: number;
  gender: number;
  embedding: Float32Array | null; // SOMA-R: external ReID embedding, L2-normalized
  headBox: Box | null;
  headDir: [number, number] | null;
  headDirConf: number;
  crowding: number;
}

// One emitted output row: [frame, tid, x, y, w, h, score]
export interface TrackRow {
  frame: number;
  tid: number;
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  ghost: boolean;
}
