"""Capsule approximations for OpenArm bimanual robot.

Each link's collision geometry is approximated by a single capsule
(swept sphere): two endpoints in the LINK'S LOCAL FRAME + a radius.

Distance between two capsules = distance between the two line segments
(capsule axes) minus the sum of their radii.  This is C^1 smooth
almost everywhere and analytically differentiable -- enabling
gradient-based self-collision avoidance WITHOUT a neural network.

The values below were AUTO-FITTED from each link's collision mesh
bounding box by `scripts/analysis/auto_fit_capsules.py`:
  * capsule axis = longest extent of the mesh bounding box
  * radius      = max perpendicular distance from axis + 5mm buffer
  * endpoints   = link-local coords accounting for <collision><origin>

Frame convention: each capsule's p0, p1 are in the link frame whose
origin coincides with the URDF link origin.
"""

# Auto-fitted from URDF collision meshes via sphere-sweep min-volume search.
CAPSULES_AUTO = {
    # MANUAL: tune p0(bottom end) / p1(top end) / r(radius) here.
    # Defaults: thin vertical column covering the central torso only;
    # ignores side shoulder-mount platforms (those collisions vs each
    # arm's link0/1 are already in INTER_BLACKLIST_NAMES).
    "openarm_body_link0":         dict(p0=(-0.004,  -0.0007,  0.0),    p1=(-0.004,  -0.0007,  0.773),  r=0.1818),
    # r restored to minimum that covers 100% of mesh (was 0.0517 → 11.9% out).
    "openarm_left_link0":         dict(p0=(-0.0009, -0.0,    -0.0),    p1=(-0.0009, -0.0,     0.0625), r=0.0667),
    # r restored to minimum that covers 100% of mesh (was 0.06 → 11.2% out).
    "openarm_left_link1":         dict(p0=(-0.0373,  0.0055,  0.0531), p1=( 0.0525, -0.0024,  0.035),  r=0.0705),
    "openarm_left_link2":         dict(p0=( 0.0298, -0.0013,  0.0826), p1=(-0.0253,  0.0017, -0.039),  r=0.0556),
    "openarm_left_link3":         dict(p0=(-0.0062,  0.0002, -0.0003), p1=(-0.0062,  0.0002,  0.1823), r=0.0527),
    "openarm_left_link4":         dict(p0=(-0.0039, -0.0305, -0.0285), p1=(-0.0039, -0.0305,  0.0965), r=0.0566),
    "openarm_left_link5":         dict(p0=(-0.0011,  0.0005,  0.0023), p1=(-0.0011,  0.0005,  0.1305), r=0.0514),
    "openarm_left_link6":         dict(p0=(-0.0351, -0.028,   0.0),    p1=(-0.0351,  0.038,   0.0),    r=0.0452),
    # r restored to minimum that covers 100% of mesh (was 0.0333 → 18.8% out).
    "openarm_left_link7":         dict(p0=( 0.0,    -0.0453, -0.0113), p1=( 0.0,     0.019,   0.1),    r=0.0384),
    "openarm_right_link0":        dict(p0=(-0.0009,  0.0,    -0.0),    p1=(-0.0009,  0.0,     0.0625), r=0.0667),
    "openarm_right_link1":        dict(p0=(-0.0373, -0.0055,  0.0531), p1=( 0.0525,  0.0024,  0.035),  r=0.0705),
    "openarm_right_link2":        dict(p0=( 0.0298,  0.0013,  0.0826), p1=(-0.0253, -0.0017, -0.039),  r=0.0556),
    "openarm_right_link3":        dict(p0=(-0.0062, -0.0002, -0.0003), p1=(-0.0062, -0.0002,  0.1823), r=0.0527),
    "openarm_right_link4":        dict(p0=(-0.0039, -0.0305, -0.0285), p1=(-0.0039, -0.0305,  0.0965), r=0.0566),
    "openarm_right_link5":        dict(p0=(-0.0011, -0.0005,  0.0023), p1=(-0.0011, -0.0005,  0.1305), r=0.0514),
    "openarm_right_link6":        dict(p0=(-0.0351, -0.038,   0.0),    p1=(-0.0351,  0.028,   0.0),    r=0.0452),
    "openarm_right_link7":        dict(p0=( 0.0,     0.0453, -0.0113), p1=( 0.0,    -0.019,   0.1),    r=0.0384),
    # r restored: hand/fingers needed much larger r to cover meshes.
    "openarm_left_hand":          dict(p0=( 0.0,    -0.0741,  0.006),  p1=( 0.0,     0.0741,  0.006),  r=0.0320),
    "openarm_left_left_finger":   dict(p0=( 0.0022,  0.0099, -0.0145), p1=( 0.0022,  0.0099,  0.0804), r=0.0375),
    "openarm_left_right_finger":  dict(p0=( 0.0022, -0.0099, -0.0145), p1=( 0.0022, -0.0099,  0.0804), r=0.0375),
    "openarm_right_hand":         dict(p0=( 0.0,    -0.0741,  0.006),  p1=( 0.0,     0.0741,  0.006),  r=0.0320),
    "openarm_right_left_finger":  dict(p0=( 0.0022,  0.0099, -0.0145), p1=( 0.0022,  0.0099,  0.0804), r=0.0375),
    "openarm_right_right_finger": dict(p0=( 0.0022, -0.0099, -0.0145), p1=( 0.0022, -0.0099,  0.0804), r=0.0375),
}


def all_capsule_definitions():
    """Return dict {link_name -> dict(p0, p1, r)} for entire dual-arm robot."""
    return dict(CAPSULES_AUTO)


# ---- Pair filter ----
# Pairs that physically can never collide (parent-child along chain,
# fixed-jointed, parallel-mounted fingers) should be skipped.

INTRA_BLACKLIST_SUFFIXES = [
    # adjacent links along the chain (kinematically parent-child)
    ("link0", "link1"), ("link1", "link2"), ("link2", "link3"),
    ("link3", "link4"), ("link4", "link5"), ("link5", "link6"),
    ("link6", "link7"), ("link7", "hand"),
    ("hand", "left_finger"), ("hand", "right_finger"),
    # gripper two fingers always parallel
    ("left_finger", "right_finger"),
    # 2-apart-in-chain pairs that structurally overlap because the
    # intervening link (elbow/wrist) is short. Verified by
    # debug_capsule_pairs.py: dist < 0 in 100% of random poses.
    ("link0", "link2"),
    ("link1", "link3"),
    ("link2", "link4"),
    ("link3", "link5"),
    ("link5", "link7"),
    # link7 vs each finger (hand is short, link7 tip overlaps fingers)
    ("link7", "left_finger"),
    ("link7", "right_finger"),
    # 3-apart: link0 (shoulder mount) and link3 (upper arm) capsules
    # geometrically cross when arm folds, even though mesh doesn't.
    # Cause of 85% of false positives. Verified by optimize_capsules.py.
    ("link0", "link3"),
]

INTER_BLACKLIST_NAMES = [
    # body vs each arm's mount (link0 fixed-jointed to body)
    ("openarm_body_link0", "openarm_left_link0"),
    ("openarm_body_link0", "openarm_right_link0"),
    ("openarm_body_link0", "openarm_left_link1"),
    ("openarm_body_link0", "openarm_right_link1"),
    # Fat body capsule (r=0.18) covers shoulder flanges and always
    # overlaps arm link2/link3 capsules when arm folds inward, even
    # though arm mesh stays clear of torso mesh. 99% of FP attributed
    # to these pairs by optimize_capsules.py.
    ("openarm_body_link0", "openarm_left_link2"),
    ("openarm_body_link0", "openarm_right_link2"),
    ("openarm_body_link0", "openarm_left_link3"),
    ("openarm_body_link0", "openarm_right_link3"),
    ("openarm_body_link0", "openarm_left_link4"),
    ("openarm_body_link0", "openarm_right_link4"),
    ("openarm_body_link0", "openarm_left_link5"),
    ("openarm_body_link0", "openarm_right_link5"),
    ("openarm_body_link0", "openarm_left_link6"),
    ("openarm_body_link0", "openarm_right_link6"),
    ("openarm_body_link0", "openarm_left_link7"),
    ("openarm_body_link0", "openarm_right_link7"),
    ("openarm_body_link0", "openarm_left_hand"),
    ("openarm_body_link0", "openarm_right_hand"),
    ("openarm_body_link0", "openarm_left_left_finger"),
    ("openarm_body_link0", "openarm_left_right_finger"),
    ("openarm_body_link0", "openarm_right_left_finger"),
    ("openarm_body_link0", "openarm_right_right_finger"),
    # The two arm bases are mounted ~12 cm apart on body, and their
    # link0 capsules (r=0.05-0.07 each) overlap regardless of pose.
    ("openarm_left_link0", "openarm_right_link0"),
]


def build_blacklist_pairs():
    """Return set of frozenset({nameA, nameB}) pairs to skip."""
    bl = set()
    for side in ("left", "right"):
        for a, b in INTRA_BLACKLIST_SUFFIXES:
            # finger suffixes are "left_finger"/"right_finger"; prefix is "openarm_{side}_"
            na = f"openarm_{side}_{a}"
            nb = f"openarm_{side}_{b}"
            bl.add(frozenset((na, nb)))
    for na, nb in INTER_BLACKLIST_NAMES:
        bl.add(frozenset((na, nb)))
    return bl
