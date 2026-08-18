// DEIMv2-Wholebody49 class vocabulary and skeleton topology.
// Direct port of soma/constants.py (python package).

export const CLASS_NAMES = [
  'body', 'adult', 'child', 'male', 'female',
  'body_with_wheelchair', 'body_with_crutches', 'head',
  'front', 'right_front', 'right_side', 'right_back', 'back',
  'left_back', 'left_side', 'left_front',
  'face', 'eye', 'nose', 'mouth', 'ear',
  'collarbone', 'shoulder', 'shoulder_left', 'shoulder_right',
  'solar_plexus', 'elbow', 'elbow_left', 'elbow_right',
  'wrist', 'wrist_left', 'wrist_right',
  'hand', 'hand_left', 'hand_right',
  'abdomen', 'hip_joint', 'hip_joint_left', 'hip_joint_right',
  'knee', 'knee_left', 'knee_right',
  'ankle', 'ankle_left', 'ankle_right',
  'foot', 'foot_left', 'foot_right',
  'bone',
] as const;

export const BODY = 0;
export const BODY_WHEELCHAIR = 5;
export const BODY_CRUTCHES = 6;
export const HEAD = 7;
export const BONE = 48;

export const BODY_ANCHOR_CLASS_IDS = [BODY, BODY_WHEELCHAIR, BODY_CRUTCHES] as const;

// Attribute classes share their bounding box with a parent class instance.
export const GENERATION_ATTRS = new Map<number, number>([[1, 0], [2, 1]]); // adult / child
export const GENDER_ATTRS = new Map<number, number>([[3, 0], [4, 1]]);     // male / female
export const HEAD_POSE_ATTRS = new Map<number, number>([
  [8, 0], [9, 1], [10, 2], [11, 3], [12, 4], [13, 5], [14, 6], [15, 7],
]);

// Keypoint classes: the box CENTER is the joint location.
export const KEYPOINT_PARENT_CLASS_IDS = [21, 22, 25, 26, 29, 35, 36, 39, 42] as const;

// parent keypoint/object class -> [left child, right child]; children share coords.
export const SIDE_PARENT_TO_CHILDREN = new Map<number, [number, number]>([
  [22, [23, 24]], // shoulder
  [26, [27, 28]], // elbow
  [29, [30, 31]], // wrist
  [32, [33, 34]], // hand
  [36, [37, 38]], // hip_joint
  [39, [40, 41]], // knee
  [42, [43, 44]], // ankle
  [45, [46, 47]], // foot
]);

// Bone(48) boxes: the box DIAGONAL always joins the centers of exactly two
// joint-class detections. Vocabulary of joinable (classid, classid) pairs.
export const BONE_EDGE_PAIRS: ReadonlyArray<[number, number]> = [
  [21, 23], [21, 24], [21, 25], [25, 35],
  [23, 27], [27, 30], [24, 28], [28, 31],
  [35, 37], [37, 40], [40, 43], [39, 42],
  [35, 38], [38, 41], [41, 44],
];

// Score/NMS defaults.
export const DEFAULT_SCORE_THRESHOLD = 0.35;
export const DEFAULT_ATTR_SCORE_THRESHOLD = 0.2;
export const DEFAULT_NMS_IOU = 0.55;
// Attribute/child boxes share coordinates with their parent: match by center
// distance (px in frame coords) + IoU.
export const ATTR_MERGE_CENTER_DIST = 12.0;

// Wholebody28(-Refine) class id -> Wholebody49 class id. 0..20 identical
// (body..ear incl. the 8 head-orientation classes); the remaining 7 parts map
// onto the same WB49 parents the WB34 table uses for those names.
export const WB28_TO_WB49: readonly number[] = (() => {
  const m = Array.from({ length: 49 }, (_v, i) => i);
  const pairs: Array<[number, number]> = [
    [21, 22], [22, 26], [23, 32], [24, 33], [25, 34], [26, 39], [27, 45],
  ];
  for (const [a, b] of pairs) {
    m[a] = b;
  }
  return m;
})();

// Wholebody25 class id -> Wholebody49 class id. 0..20 identical; wb25 lacks
// the shoulder/elbow/knee keypoints — its tail is hand(21) + L/R and foot(24).
export const WB25_TO_WB49: readonly number[] = (() => {
  const m = Array.from({ length: 49 }, (_v, i) => i);
  const pairs: Array<[number, number]> = [
    [21, 32], [22, 33], [23, 34], [24, 45],
  ];
  for (const [a, b] of pairs) {
    m[a] = b;
  }
  return m;
})();

// Classes the active tracking stack actually consumes (soma/perception.py
// LEAN_CLASS_IDS): body anchors, attributes, head + orientation ring, and the
// keypoint vocabulary incl. L/R children. face parts and hand/foot objects
// feed nothing.
export const LEAN_CLASS_IDS: ReadonlySet<number> = new Set([
  ...Array.from({ length: 16 }, (_v, i) => i),        // 0..15
  ...Array.from({ length: 11 }, (_v, i) => 21 + i),   // 21..31
  ...Array.from({ length: 10 }, (_v, i) => 35 + i),   // 35..44
]);
