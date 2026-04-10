"""OpenArm-specific constants and utility re-exports for VPP-TC.

All dual-arm constants are ordered: left arm (7) + right arm (7) = 14.
"""

from vpptc.utils import compute_min_center_distance, compute_qe, feasible_qdd_region

# Per-arm acceleration limits (derived from peak torque / mass matrix, ~80% margin)
OPENARM_JOINT_ACCELERATION_LIMITS = [
    (-70.0, 70.0),    # J1: DM8009P, α_peak≈89
    (-80.0, 80.0),    # J2: DM8009P, α_peak≈102
    (-80.0, 80.0),    # J3: DM4340,  α_peak≈108
    (-75.0, 75.0),    # J4: DM4340,  α_peak≈94
    (-120.0, 120.0),  # J5: DM4310,  α_peak config-dependent
    (-120.0, 120.0),  # J6: DM4310,  α_peak≈159
    (-120.0, 120.0),  # J7: DM4310,  α_peak≈163
]

# Per-arm velocity limits (from joint_limits.yaml)
OPENARM_VELOCITY_LIMITS = [16.754666, 16.754666, 5.445426, 5.445426,
                           20.943946, 20.943946, 20.943946]

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

# Home configurations (arms up, away from body, symmetric)
Q_HOME_RIGHT = [1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
Q_HOME_LEFT = [-1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
Q_HOME_DUAL = Q_HOME_LEFT + Q_HOME_RIGHT

N_DOF = 14  # 7 per arm × 2
