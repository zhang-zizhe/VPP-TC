"""OpenArm-specific constants and utility re-exports for VPP-TC.

All dual-arm constants are ordered: left arm (7) + right arm (7) = 14.
"""

from vpptc.utils import compute_min_center_distance, compute_qe, feasible_qdd_region

# Per-arm acceleration limits.
# 2026-06-10: derated from a PEAK-torque basis to CONTINUOUS (nominal) torque.
# Braking-to-a-stop (qe / viability) needs SUSTAINED deceleration through the
# whole stop, not a brief thermal-limited peak, so the sustainable a_max is
#   a_max = a_max_peak * (tau_nominal / tau_peak)
# with the Damiao nominal/peak torques (verified from datasheets):
#   DM8009P (J1,J2): 20/40 = 0.50   ->  70->35, 80->40
#   DM4340  (J3,J4):  9/27 = 0.33   ->  80->27, 75->25
#   DM4310  (J5-J7):  3/7  = 0.43   -> 120->51
# NOTE: still ignores reflected rotor inertia (N^2 * J_rotor) -- would lower
# J3/J4 (40:1) further -- and gravity headroom.  So these are an UPPER bound on
# the sustainable a_max; conservative but not yet exact.  See
# docs/ACCELERATION_LIMITS.md.  Legacy peak-basis values were
#   [70, 80, 80, 75, 120, 120, 120]  (optimistic).
OPENARM_JOINT_ACCELERATION_LIMITS = [
    (-9.0,  9.0),    # J1
    (-9.7,  9.7),    # J2
    (-9.3,  9.3),    # J3
    (-5.6,  5.6),    # J4
    (-20.3, 20.3),   # J5
    (-23.0, 23.0),   # J6
    (-12.0, 12.0),   # J7
]
# 2026-06-10 (v2): grounded a_max = (tau_continuous - |tau_gravity|) / J_effective,
# evaluated over 2000 random configs, 5th-percentile (worst-case) per joint.
# Accounts for: continuous (not peak) torque, gravity headroom, worst-case
# configuration inertia (from the URDF mass matrix).  Comes out ~Panda-class.
# Still ignores reflected rotor inertia (40:1 J3/J4) and joint friction -- both
# would lower it slightly more, so these are a (mild) upper bound.
#   history: peak-basis [70,80,80,75,120,120,120] -> continuous [35,40,27,25,51,51,51]
#            -> this (grav-adjusted, worst-case).  See docs/ACCELERATION_LIMITS.md.

# Per-arm velocity limits = 0.12 x the URDF no-load max speeds.
# 2026-06-12: anchored to Panda's validated working regime.  0.12x gives a
# per-arm clamp (qe-overshoots-a-joint-limit) rate of ~10.5%, matching Panda's
# 10.6% -- the regime where VPP-TC's viability term is informative but not
# saturated at the joint limits.  0.15x drifted to 16.5%/arm (above Panda);
# 0.12x = ~0.76x Panda speed, which compensates OpenArm's narrower joint range
# (~180 deg vs Panda's ~332 deg).  Controller velocity limit MUST track this
# same constant (in-distribution).  See clamp-rate grid analysis.
OPENARM_VELOCITY_LIMITS = [2.011, 2.011, 0.653, 0.653,
                           2.513, 2.513, 2.513]   # 0.12x no-load (~Panda regime)
# Legacy no-load max (URDF/official): [16.754666, 16.754666, 5.445426,
#   5.445426, 20.943946, 20.943946, 20.943946]; rated: ~[10.5,10.5,3.77,3.77,12.6,12.6,12.6]

# Per-arm effort limits
OPENARM_EFFORT_LIMITS = [40, 40, 27, 27, 7, 7, 7]

# Right arm joint limits (from bimanual URDF, reflect=1)
OPENARM_Q_MIN_RIGHT = [-1.396263, -0.174533, -1.570796, 0.0,
                       -1.570796, -0.785398, -1.570796]
OPENARM_Q_MAX_RIGHT = [3.490659, 3.316125, 1.570796, 2.443461,
                       1.570796, 0.785398, 1.570796]

# Left arm joint limits (from bimanual URDF, reflect=-1 + offsets)
OPENARM_Q_MIN_LEFT = [-3.490659, -3.316125, -1.570796, 0.0,
                      -1.570796, -0.785398, -1.570796]
OPENARM_Q_MAX_LEFT = [1.396263, 0.174533, 1.570796, 2.443461,
                      1.570796, 0.785398, 1.570796]

# Dual-arm constants (left + right = 14 elements)
DUAL_ACCELERATION_LIMITS = OPENARM_JOINT_ACCELERATION_LIMITS + OPENARM_JOINT_ACCELERATION_LIMITS
DUAL_VELOCITY_LIMITS = OPENARM_VELOCITY_LIMITS + OPENARM_VELOCITY_LIMITS
DUAL_Q_MIN = OPENARM_Q_MIN_LEFT + OPENARM_Q_MIN_RIGHT
DUAL_Q_MAX = OPENARM_Q_MAX_LEFT + OPENARM_Q_MAX_RIGHT

# Per-joint (lower, upper) position limits, 14 elements -- pass to
# compute_qe(..., pos_limits=DUAL_POS_LIMITS) so the braking stop pose qe is
# clamped to the joint range (an un-clamped qe is non-physical; ~92% of the
# legacy 8M qe poses violated a joint limit).
DUAL_POS_LIMITS = list(zip(DUAL_Q_MIN, DUAL_Q_MAX))

# Home configurations (arms up, away from body, symmetric)
Q_HOME_RIGHT = [1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
Q_HOME_LEFT = [-1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
Q_HOME_DUAL = Q_HOME_LEFT + Q_HOME_RIGHT

N_DOF = 14  # 7 per arm × 2
