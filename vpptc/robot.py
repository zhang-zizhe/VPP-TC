"""Franka Emika Panda robot wrapper for PyBullet simulation."""

import os
from typing import List, Tuple

import numpy as np
import pybullet as p

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_URDF_DIR = os.path.join(_PACKAGE_DIR, os.pardir, "assets", "urdf")


class Panda:
    """PyBullet wrapper for the 7-DOF Franka Emika Panda manipulator.

    Parameters
    ----------
    stepsize : float
        Simulation time step in seconds.
    realtime : int
        Whether to run in real-time mode (0 = off, 1 = on).
    urdf_dir : str or None
        Path to the directory containing ``panda/`` and ``plane/`` URDF
        sub-folders.  Defaults to ``<project>/assets/urdf/``.
    """

    # Default initial joint configuration
    # DEFAULT_Q0 = [0.669, 0.346, 0.5, -1.66, -0.367, 2.3, 1.99]
    DEFAULT_Q0 = [0.669, 0.546, 0.8, -1.26, -0.367, 1.3, 1.99]

    def __init__(
        self,
        stepsize: float = 1e-3,
        realtime: int = 0,
        urdf_dir: str = None,
    ):
        self.t: float = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque"
        self.position_control_gain_p = [0.1] * 7
        self.position_control_gain_d = [1.0] * 7
        self.max_torque = [10000] * 7

        # Camera defaults
        self.cam_base_yaw = 30
        self.cam_pitch = -20
        self.cam_dist = 1.0
        self.cam_target = [0, 0, 0.5]

        if urdf_dir is None:
            urdf_dir = _DEFAULT_URDF_DIR

        # --- PyBullet initialisation ---
        p.connect(p.GUI, options="--width=1280 --height=720")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=self.cam_dist,
            cameraYaw=self.cam_base_yaw,
            cameraPitch=self.cam_pitch,
            cameraTargetPosition=self.cam_target,
        )
        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0, 0, 0)

        # Load ground plane
        plane_urdf = os.path.join(urdf_dir, "plane", "plane.urdf")
        self.plane = p.loadURDF(plane_urdf, useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=0.95)

        # Load Panda robot with self-collision enabled
        panda_urdf = os.path.join(urdf_dir, "panda", "panda.urdf")
        self.robot = p.loadURDF(
            panda_urdf,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)

        # Exclude the virtual fixed joint (flange <-> last link)
        self.dof = p.getNumJoints(self.robot) - 1

        self.joints: List[int] = []
        self.q_min: List[float] = []
        self.q_max: List[float] = []
        self.target_pos: List[float] = []
        self.target_torque: List[float] = []

        for j in range(self.dof):
            info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(info[8])
            self.q_max.append(info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j]) / 2.0)
            self.target_torque.append(0.0)

        self.reset()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the robot to its default initial configuration."""
        self.t = 0.0
        self.control_mode = "torque"
        self.target_pos = list(self.DEFAULT_Q0)
        for j in range(self.dof):
            self.target_torque[j] = 0.0
            p.resetJointState(self.robot, j, targetValue=self.target_pos[j])
        self.resetController()

    def step(self) -> None:
        """Advance the simulation by one time step."""
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self) -> None:
        """Disable the default velocity controller (required for torque mode)."""
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * self.dof,
        )

    # ------------------------------------------------------------------
    # Control modes
    # ------------------------------------------------------------------

    def setControlMode(self, mode: str) -> None:
        """Switch control mode ('position', 'velocity', or 'torque')."""
        if mode == "position":
            self.control_mode = "position"
        elif mode == "velocity":
            self.control_mode = "velocity"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise ValueError(f"Unknown control mode: {mode!r}")

    def setTargetPositions(self, target_pos) -> None:
        """Send a position command to the robot."""
        self.target_pos = target_pos
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_pos,
        )

    def setTargetTorques(self, target_torque) -> None:
        """Send a torque command to the robot."""
        self.target_torque = target_torque
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.TORQUE_CONTROL,
            forces=self.target_torque,
        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def getJointStates(self) -> Tuple[List[float], List[float]]:
        """Return current joint positions and velocities."""
        states = p.getJointStates(self.robot, self.joints)
        positions = [s[0] for s in states]
        velocities = [s[1] for s in states]
        return positions, velocities

    def solveInverseDynamics(self, pos, vel, acc) -> List[float]:
        """Compute inverse dynamics torques for given (pos, vel, acc)."""
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori) -> List[float]:
        """Compute inverse kinematics for the end-effector (link 7)."""
        return list(p.calculateInverseKinematics(self.robot, 7, pos, ori))

    def solveForwardKinematics(self) -> Tuple:
        """Return end-effector position and orientation."""
        pos = p.getLinkState(self.robot, 7)[0]
        ori = p.getLinkState(self.robot, 7)[1]
        return pos, ori

    def getEndVelocity(self):
        """Return the linear velocity of the end-effector."""
        return p.getLinkState(self.robot, 7, True)[6]

    def getEndAngularVelocity(self):
        """Return the angular velocity of the end-effector."""
        return p.getLinkState(self.robot, 7, True)[7]

    def applyForce(self, force) -> None:
        """Apply an external force at the end-effector."""
        ee_pos = p.getLinkState(self.robot, 7)[0]
        p.applyExternalForce(self.robot, 7, force, ee_pos, p.WORLD_FRAME)

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------

    def getJacobian(self):
        """Return the linear Jacobian of the end-effector (3 x 7)."""
        joint_pos = [p.getJointState(self.robot, j)[0] for j in range(7)]
        zeros = [0.0] * 7
        return p.calculateJacobian(self.robot, 7, [0, 0, 0], joint_pos, zeros, zeros)[0]

    def getJacobian_ori(self):
        """Return the angular Jacobian of the end-effector (3 x 7)."""
        joint_pos = [p.getJointState(self.robot, j)[0] for j in range(7)]
        zeros = [0.0] * 7
        return p.calculateJacobian(self.robot, 7, [0, 0, 0], joint_pos, zeros, zeros)[1]

    def getMassMatrix(self, joint_states):
        """Return the joint-space mass matrix."""
        return p.calculateMassMatrix(self.robot, joint_states)

    def getClosestPoints(self, link1: int, link2: int):
        """Return the closest points between two links of the robot."""
        return p.getClosestPoints(self.robot, self.robot, 1000, link1, link2)
