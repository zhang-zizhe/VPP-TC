import pybullet as p
import pybullet_data
import time

cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF(
    "assets/urdf/panda/panda_dual_arms.urdf",
    useFixedBase=True,
)
p.setGravity(0, 0, -9.81)

while True:
    p.stepSimulation()
    time.sleep(1 / 240)