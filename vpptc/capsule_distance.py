"""Analytic capsule-capsule distance with gradients.

A capsule = swept sphere along a line segment [p0, p1] with radius r.
Distance between two capsules = segment_segment_distance(seg_a, seg_b)
                                - r_a - r_b.

Provides two implementations:
  * NumPy version (fast, used for data sampling).
  * PyTorch version (differentiable, used in SCA QP for runtime).

Reference algorithm: D. Eberly, "Robust Computation of Distance
Between Line Segments", Geometric Tools.
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

EPS = 1e-12


# =====================================================================
# NumPy version
# =====================================================================

def segment_segment_distance_np(p1, p2, p3, p4):
    """Min distance and closest points between segments [p1,p2] and [p3,p4].

    All inputs: shape (3,) arrays.
    Returns (dist, c_on_ab, c_on_cd).
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p4 = np.asarray(p4, dtype=np.float64)

    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = u @ u
    b = u @ v
    c = v @ v
    d = u @ w
    e = v @ w
    D = a * c - b * b

    if D < EPS:
        # Parallel: pick t to project p1 onto cd, then clamp.
        sN, sD = 0.0, 1.0
        tN, tD = e, c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        sD = D
        tD = D
        if sN < 0.0:
            sN = 0.0; tN = e; tD = c
        elif sN > sD:
            sN = sD; tN = e + b; tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b; sD = a

    sc = 0.0 if abs(sN) < EPS else sN / sD
    tc = 0.0 if abs(tN) < EPS else tN / tD

    c1 = p1 + sc * u
    c2 = p3 + tc * v
    dist = np.linalg.norm(c1 - c2)
    return dist, c1, c2


def capsule_capsule_distance_np(cap_a, cap_b):
    """Distance between two capsules in WORLD frame.

    cap_a, cap_b: dict with keys 'p0', 'p1', 'r' (all in world coords).
    Returns signed distance (negative = interpenetration).
    """
    seg_d, _, _ = segment_segment_distance_np(
        cap_a['p0'], cap_a['p1'], cap_b['p0'], cap_b['p1'])
    return seg_d - cap_a['r'] - cap_b['r']


# =====================================================================
# Torch version (differentiable)
# =====================================================================

if HAS_TORCH:
    def segment_segment_distance_torch(p1, p2, p3, p4):
        """Differentiable segment-segment distance.

        All inputs: torch tensors shape (3,) or (B, 3) batched.
        Returns dist tensor (scalar or (B,)).
        """
        u = p2 - p1
        v = p4 - p3
        w = p1 - p3
        a = (u * u).sum(-1)
        b = (u * v).sum(-1)
        c = (v * v).sum(-1)
        d = (u * w).sum(-1)
        e = (v * w).sum(-1)
        D = a * c - b * b

        # The branching algorithm is hard to vectorize cleanly.
        # Use a smooth approximation: compute sc, tc by closed form
        # (assuming non-parallel), then clamp to [0, 1].
        # This loses some accuracy near edge cases but stays differentiable.
        D_safe = torch.where(D > EPS, D, torch.full_like(D, EPS))
        sc = (b * e - c * d) / D_safe
        tc = (a * e - b * d) / D_safe

        # Clamp to segment interiors (loses gradient at clamp boundary
        # but that's a measure-zero event).
        sc = sc.clamp(0.0, 1.0)
        tc = tc.clamp(0.0, 1.0)

        # Recompute tc given clamped sc (one-sided refinement)
        # tc = (sc * b + e) / c   when projecting onto segment cd
        c_safe = torch.where(c > EPS, c, torch.full_like(c, EPS))
        tc_refined = ((sc * b - (-e)) / c_safe).clamp(0.0, 1.0)
        # Recompute sc given refined tc
        a_safe = torch.where(a > EPS, a, torch.full_like(a, EPS))
        sc_refined = ((tc_refined * b - d) / a_safe).clamp(0.0, 1.0)

        c1 = p1 + sc_refined.unsqueeze(-1) * u
        c2 = p3 + tc_refined.unsqueeze(-1) * v
        return torch.linalg.norm(c1 - c2, dim=-1)


    def capsule_capsule_distance_torch(p0a, p1a, ra, p0b, p1b, rb):
        """Differentiable capsule-capsule distance.

        All endpoints in world frame, all torch tensors.
        ra, rb scalar tensors or floats.
        """
        seg_d = segment_segment_distance_torch(p0a, p1a, p0b, p1b)
        return seg_d - ra - rb


# =====================================================================
# Local -> world transform helpers
# =====================================================================

def transform_capsule(cap_local, link_pos, link_orn_mat):
    """Transform a capsule from link-local frame to world frame.

    cap_local: dict with 'p0', 'p1' (tuples/arrays in link frame), 'r'.
    link_pos:  (3,) world position of link frame origin.
    link_orn_mat: (3,3) rotation matrix link->world.
    Returns dict {'p0', 'p1', 'r'} in world coords.
    """
    p0_local = np.asarray(cap_local['p0'], dtype=np.float64)
    p1_local = np.asarray(cap_local['p1'], dtype=np.float64)
    link_pos = np.asarray(link_pos, dtype=np.float64)
    R = np.asarray(link_orn_mat, dtype=np.float64).reshape(3, 3)
    return {
        'p0': link_pos + R @ p0_local,
        'p1': link_pos + R @ p1_local,
        'r':  cap_local['r'],
    }


if __name__ == "__main__":
    # Quick sanity check
    cap1 = dict(p0=(0,0,0), p1=(0,0,1), r=0.1)
    cap2 = dict(p0=(0.3,0,0), p1=(0.3,0,1), r=0.1)
    d = capsule_capsule_distance_np(cap1, cap2)
    print(f"Two parallel capsules 0.3m apart, r=0.1 each: dist={d:.4f} (expect 0.1)")

    cap3 = dict(p0=(0,0,0), p1=(0,0,1), r=0.1)
    cap4 = dict(p0=(0,0,0), p1=(0,0,1), r=0.1)
    d = capsule_capsule_distance_np(cap3, cap4)
    print(f"Two coincident capsules: dist={d:.4f} (expect -0.2)")

    cap5 = dict(p0=(0,0,0), p1=(1,0,0), r=0.05)
    cap6 = dict(p0=(0.5,0.3,0), p1=(0.5,0.3,1), r=0.05)
    d = capsule_capsule_distance_np(cap5, cap6)
    print(f"Crossed capsules at (0.5, 0.3): dist={d:.4f} (expect 0.2)")
