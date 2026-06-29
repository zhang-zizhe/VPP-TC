#!/usr/bin/env python3
"""Inspect convex-hull over-detection in a 3D GUI by "feeling" it.

Overlay each link's [convex hull] (the shape the collision engine actually uses) in
semi-transparent red on top of the real robot.
Drag the 14 joint sliders on the right to bring the two arms together -- you will see:
  the red convex hulls collide first (the engine reports "collision"), but the gray
  [real meshes] are still separated by empty space.
That is over-detection: the convex hull of a concave link (hand/link5/link4/...)
fills in the empty concave region as solid.

Run with:  python3 scripts/visualize_convex_hulls.py
Drag the mouse to rotate the view; scroll to zoom.
"""
import os, re, time, glob
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
import pybullet as p
import pybullet_data

URDF = 'assets/urdf/openarm_description/urdf/robot/openarm_bimanual.urdf'
UDIR = os.path.dirname(URDF)
HULLDIR = 'scratch_hulls'
DECOMP = {  # already fixed via V-HACD, overlaid in green for comparison
    'openarm_body_link0': 'scratch_vhacd/decomp_res4000000.obj',
}

# ---- 1) Parse URDF, build a convex-hull obj for each link (link-local, meters; with signed scale + origin) ----
def _T(xyz, rpy):
    T = trimesh.transformations.euler_matrix(*rpy); T[:3, 3] = xyz; return T

def build_hulls():
    os.makedirs(HULLDIR, exist_ok=True)
    root = ET.parse(URDF).getroot(); built = {}
    for link in root.findall('link'):
        nm = link.get('name'); col = link.find('collision')
        if col is None: continue
        mesh = col.find('geometry/mesh')
        if mesh is None: continue
        fp = os.path.normpath(os.path.join(UDIR, mesh.get('filename')))
        if not os.path.exists(fp): continue
        sc = [float(x) for x in mesh.get('scale', '1 1 1').split()]
        o = col.find('origin')
        xyz = [float(x) for x in (o.get('xyz', '0 0 0').split() if o is not None else '0 0 0'.split())]
        rpy = [float(x) for x in (o.get('rpy', '0 0 0').split() if o is not None else '0 0 0'.split())]
        m = trimesh.load(fp)
        m.apply_transform(np.diag(sc + [1.0])); m.apply_transform(_T(xyz, rpy))
        outp = f'{HULLDIR}/{nm}.obj'
        m.convex_hull.export(outp)
        ratio = m.convex_hull.volume / max(m.volume, 1e-9)
        built[nm] = (outp, ratio)
    return built

print('building convex hulls ...')
hulls = build_hulls()

# ---- 2) GUI ----
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(1.4, 50, -25, [0.3, 0.0, 0.7])
r = p.loadURDF(URDF, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

i2n, arm, armlinks = {}, [], []
for i in range(p.getNumJoints(r)):
    info = p.getJointInfo(r, i); nm = info[12].decode()
    i2n[i] = nm
    if info[2] == 0 and re.match(r'openarm_(left|right)_joint[1-7]$', info[1].decode()):
        arm.append(i)
    if re.match(r'openarm_(left|right)_(link[1-7]|hand|.*finger)$', nm):
        armlinks.append(i)

# convex-hull overlay bodies (semi-transparent red, visual only, more concave = redder)
# links already fixed via V-HACD skip the red convex hull (their "convex hull" would collapse back into one big hull, which is misleading); use the green decomposition instead to show the real collision shape
hull_bodies = {}
for i, nm in i2n.items():
    if nm not in hulls or nm in DECOMP: continue
    ratio = hulls[nm][1]
    a = 0.45 if ratio > 1.5 else 0.18          # make concave ones more prominent
    vs = p.createVisualShape(p.GEOM_MESH, fileName=hulls[nm][0], rgbaColor=[1, 0.2, 0.2, a])
    hull_bodies[i] = p.createMultiBody(0, baseVisualShapeIndex=vs)
# green: the fixed V-HACD decomposition (comparison)
decomp_bodies = {}
for i, nm in i2n.items():
    if nm in DECOMP and os.path.exists(DECOMP[nm]):
        vs = p.createVisualShape(p.GEOM_MESH, fileName=DECOMP[nm], meshScale=[0.001]*3, rgbaColor=[0.2, 0.9, 0.3, 0.4])
        decomp_bodies[i] = p.createMultiBody(0, baseVisualShapeIndex=vs)

# find a "convex hulls collide but visually nothing touches" pose to use as the default (sample and pick the one with the closest cross-arm convex hulls)
print('searching a near-collision pose ...')
lims = [(p.getJointInfo(r, j)[8], p.getJointInfo(r, j)[9]) for j in arm]
rng = np.random.default_rng(0); best_q, best_d = [0.0]*len(arm), 9.9
left = [i for i in armlinks if 'left' in i2n[i]]; right = [i for i in armlinks if 'right' in i2n[i]]
for _ in range(4000):
    q = [rng.uniform(lo, hi)*0.7 for lo, hi in lims]
    for j, v in zip(arm, q): p.resetJointState(r, j, v)
    p.performCollisionDetection()
    d = 9.9
    for la in left:
        for ra in right:
            cps = p.getClosestPoints(r, r, 0.05, la, ra)
            if cps: d = min(d, min(c[8] for c in cps))
    if d < best_d: best_d, best_q = d, q

sliders = [p.addUserDebugParameter(i2n[j].replace('openarm_', ''), lims[k][0], lims[k][1], best_q[k])
           for k, j in enumerate(arm)]
toggle = p.addUserDebugParameter('  >>> drag this=0 to hide red convex hulls', 0, 1, 1)

print('\n===== controls =====')
print(' drag mouse to rotate, scroll to zoom')
print(' the 14 sliders on the right = 14 arm joints; drag to bring the two arms together')
print(' red = convex hull (used by engine collisions)  gray = real mesh  green = the fixed V-HACD decomposition of body0')
print(' watch: when the red convex hulls collide, the gray real meshes are often still separated by empty space = over-detection\n')

txt = None
while p.isConnected():
    for j, s in zip(arm, sliders):
        p.resetJointState(r, j, p.readUserDebugParameter(s))
    show = p.readUserDebugParameter(toggle) > 0.5
    for i, hb in hull_bodies.items():
        ls = p.getLinkState(r, i)
        p.resetBasePositionAndOrientation(hb, ls[4], ls[5])
        p.changeVisualShape(hb, -1, rgbaColor=[1, 0.2, 0.2, (0.45 if hulls[i2n[i]][1] > 1.5 else 0.18) if show else 0])
    for i, db in decomp_bodies.items():
        ls = p.getLinkState(r, i); p.resetBasePositionAndOrientation(db, ls[4], ls[5])
    # real-time closest cross-arm convex-hull distance
    p.performCollisionDetection()
    d = 9.9; pair = ''
    for la in left:
        for ra in right:
            cps = p.getClosestPoints(r, r, 0.3, la, ra)
            if cps:
                mc = min(cps, key=lambda c: c[8])
                if mc[8] < d: d, pair = mc[8], f'{i2n[la]}<->{i2n[ra]}'
    verdict = 'engine verdict: collision (but check whether the gray real meshes actually touch?)' if d < 0 else 'engine verdict: no collision'
    if txt is not None: p.removeUserDebugItem(txt)
    txt = p.addUserDebugText(f'closest cross-arm convex-hull distance: {d*1000:+.0f} mm   {verdict}   [{pair}]',
                             [0.0, 0.0, 1.35], textColorRGB=[1, 0, 0] if d < 0 else [0, 0.5, 0], textSize=1.4)
    time.sleep(1/120)
