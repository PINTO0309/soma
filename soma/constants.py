"""DEIMv2-Wholebody49 class vocabulary and skeleton topology.

Mirrors docs/488_DEIMv2-Wholebody49/classes.txt and the constants in
docs/DEIMv2/demo/wholebody49/demo_deimv2_torch_wholebody49_ins.py.
"""
from __future__ import annotations

CLASS_NAMES = (
    "body", "adult", "child", "male", "female",
    "body_with_wheelchair", "body_with_crutches", "head",
    "front", "right_front", "right_side", "right_back", "back",
    "left_back", "left_side", "left_front",
    "face", "eye", "nose", "mouth", "ear",
    "collarbone", "shoulder", "shoulder_left", "shoulder_right",
    "solar_plexus", "elbow", "elbow_left", "elbow_right",
    "wrist", "wrist_left", "wrist_right",
    "hand", "hand_left", "hand_right",
    "abdomen", "hip_joint", "hip_joint_left", "hip_joint_right",
    "knee", "knee_left", "knee_right",
    "ankle", "ankle_left", "ankle_right",
    "foot", "foot_left", "foot_right",
    "bone",
)

BODY = 0
BODY_WHEELCHAIR = 5
BODY_CRUTCHES = 6
HEAD = 7
FACE = 16
HAND = 32
FOOT = 45
BONE = 48

BODY_ANCHOR_CLASS_IDS = (BODY, BODY_WHEELCHAIR, BODY_CRUTCHES)

# Attribute classes share their bounding box with a parent class instance.
GENERATION_ATTRS = {1: 0, 2: 1}                    # adult / child
GENDER_ATTRS = {3: 0, 4: 1}                        # male / female
HEAD_POSE_ATTRS = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}
ATTRIBUTE_CLASS_IDS = frozenset(GENERATION_ATTRS) | frozenset(GENDER_ATTRS) | frozenset(HEAD_POSE_ATTRS)

# Keypoint classes: the box CENTER is the joint location.
KEYPOINT_PARENT_CLASS_IDS = (21, 22, 25, 26, 29, 35, 36, 39, 42)
KEYPOINT_CLASS_IDS = frozenset(
    {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44}
)

# parent keypoint/object class -> (left child, right child); children share coords.
SIDE_PARENT_TO_CHILDREN = {
    22: (23, 24),   # shoulder
    26: (27, 28),   # elbow
    29: (30, 31),   # wrist
    32: (33, 34),   # hand
    36: (37, 38),   # hip_joint
    39: (40, 41),   # knee
    42: (43, 44),   # ankle
    45: (46, 47),   # foot
}
LEFT_SIDE_CLASS_IDS = frozenset({23, 27, 30, 33, 37, 40, 43, 46})
RIGHT_SIDE_CLASS_IDS = frozenset({24, 28, 31, 34, 38, 41, 44, 47})

FACE_PART_CLASS_IDS = frozenset({17, 18, 19, 20})  # eye, nose, mouth, ear

# Bone(48) boxes: the box DIAGONAL always joins the centers of exactly two
# joint-class detections (dataset design guarantees a unique diagonal).
# Vocabulary of joinable (classid, classid) pairs, from the demo.
BONE_EDGE_PAIRS = (
    (21, 23), (21, 24), (21, 25), (25, 35),
    (23, 27), (27, 30), (24, 28), (28, 31),
    (35, 37), (37, 40), (40, 43), (39, 42),
    (35, 38), (38, 41), (41, 44),
)

# Undirected skeleton edges over PARENT keypoint ids (for somatotype limbs).
SOMATOTYPE_LIMBS = (
    (21, 22),   # collarbone-shoulder (half clavicle span)
    (21, 25),   # collarbone-solar_plexus
    (25, 35),   # solar_plexus-abdomen
    (22, 26),   # upper arm
    (26, 29),   # forearm
    (35, 36),   # abdomen-hip
    (36, 39),   # thigh
    (39, 42),   # shin
)

# Score/NMS defaults (exp006 informs the operating band).
DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_ATTR_SCORE_THRESHOLD = 0.20
DEFAULT_NMS_IOU = 0.55
# Attribute/child boxes share coordinates with their parent: match by center
# distance (px in frame coords) + IoU, as in the demo (10 px at demo scale).
ATTR_MERGE_CENTER_DIST = 12.0
