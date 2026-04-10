#!/usr/bin/env python3
import os
import time
import pybullet as p
import pybullet_data

p.connect(p.GUI, options='--width=1280 --height=720')
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=50,
    cameraPitch=-20,
    cameraTargetPosition=[0.2, -0.2, 0.8]
)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')

urdf_path = os.path.abspath('assets/urdf/openarm_description/urdf/robot/openarm_bimanual.urdf')
print('Actually loading URDF:', urdf_path)

robot = p.loadURDF(
    urdf_path,
    useFixedBase=True,
    flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
)

p.setGravity(0, 0, 0)


while p.isConnected():
    p.stepSimulation()
    time.sleep(1 / 240)