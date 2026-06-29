"""Dual-arm OpenArm robot wrapper for PyBullet simulation.

Loads the bimanual URDF (body + 2×7-DOF arms + hands).  Only the 14
revolute arm joints are controlled; hand/finger joints exist for
collision detection only.
"""

import os
import re
from typing import List, Tuple

import numpy as np
import pybullet as p

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_URDF = os.path.join(
    _PACKAGE_DIR, os.pardir, "assets", "urdf",
    "openarm_description", "urdf", "robot", "openarm_bimanual.urdf",
)


def _find_link_index(robot_id, name):
    """Return the link index whose child-link name matches *name*."""
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
            return i
    raise ValueError(f"Link {name!r} not found")


def _setup_collision_filters(robot_id):
    """Exclude only link5↔link7 pairs within each arm (permanently overlapping)."""
    link5s, link7s = [], []
    for i in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, i)[12].decode("utf-8")
        if name.endswith("_link5"):
            link5s.append(i)
        elif name.endswith("_link7"):
            link7s.append(i)
    for l5 in link5s:
        for l7 in link7s:
            if abs(l5 - l7) < 5:
                p.setCollisionFilterPair(
                    robot_id, robot_id, l5, l7, enableCollision=0)


def _set_finger_positions(robot_id, position=0.01):
    """Fix all finger joints at a given position to avoid finger-finger overlap."""
    for i in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, i)[1].decode("utf-8")
        if "finger_joint" in name:
            p.resetJointState(robot_id, i, position)


class OpenArm:
    """PyBullet wrapper for the dual-arm OpenArm manipulator.

    Parameters
    ----------
    stepsize : float
        Simulation time step in seconds.
    realtime : int
        Whether to run in real-time mode (0 = off, 1 = on).
    urdf_path : str or None
        Path to the bimanual URDF file.
    """

    from vpptc.utils_openarm import Q_HOME_DUAL as _DEFAULT_Q0

    def __init__(
        self,
        stepsize: float = 1e-3,
        realtime: int = 0,
        urdf_path: str = None,
    ):
        self.t: float = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.control_mode = "torque"

        if urdf_path is None:
            urdf_path = _DEFAULT_URDF

        # --- PyBullet initialisation ---
        p.connect(p.GUI, options="--width=1280 --height=720")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=50, cameraPitch=-20,
            cameraTargetPosition=[0.2, -0.2, 0.8],
        )
        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0, 0, 0)

        # Load robot
        self.robot = p.loadURDF(
            urdf_path, useFixedBase=True,
            flags=(p.URDF_USE_SELF_COLLISION
                   | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT),
        )
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)
        _setup_collision_filters(self.robot)
        _set_finger_positions(self.robot, 0.01)

        # Collect finger joint indices (to reset during simulation)
        self.finger_joints: List[int] = []
        for i in range(p.getNumJoints(self.robot)):
            if "finger_joint" in p.getJointInfo(self.robot, i)[1].decode("utf-8"):
                self.finger_joints.append(i)

        # --- Discover arm joints (revolute, name matches openarm_*_joint[1-7]) ---
        _arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
        self.arm_joints: List[int] = []
        self.q_min: List[float] = []
        self.q_max: List[float] = []

        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            jname = info[1].decode("utf-8")
            if info[2] == p.JOINT_REVOLUTE and _arm_re.match(jname):
                self.arm_joints.append(j)
                self.q_min.append(info[8])
                self.q_max.append(info[9])

        self.dof = len(self.arm_joints)
        assert self.dof == 14, f"Expected 14 arm DOFs, found {self.dof}"

        # --- End-effector link indices ---
        self.left_ee = _find_link_index(self.robot, "openarm_left_hand_tcp")
        self.right_ee = _find_link_index(self.robot, "openarm_right_hand_tcp")

        # --- Collect all link indices per arm (for distance checks) ---
        self.left_links: List[int] = []
        self.right_links: List[int] = []
        for i in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, i)[12].decode("utf-8")
            if name.startswith("openarm_left_"):
                self.left_links.append(i)
            elif name.startswith("openarm_right_"):
                self.right_links.append(i)

        self.target_torque = [0.0] * self.dof
        self.reset()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the robot to its default initial configuration."""
        self.t = 0.0
        self.control_mode = "torque"
        q0 = list(self._DEFAULT_Q0)
        for i, j in enumerate(self.arm_joints):
            self.target_torque[i] = 0.0
            p.resetJointState(self.robot, j, targetValue=q0[i])
        for fj in self.finger_joints:
            p.resetJointState(self.robot, fj, 0.01)
        self._resetController()

    def step(self) -> None:
        """Advance the simulation by one time step."""
        self.t += self.stepsize
        p.stepSimulation()

    def _resetController(self) -> None:
        """Disable the default velocity controller on arm joints."""
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.arm_joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * self.dof,
        )

    def setControlMode(self, mode: str) -> None:
        if mode == "torque":
            if self.control_mode != "torque":
                self._resetController()
            self.control_mode = "torque"
        elif mode in ("position", "velocity"):
            self.control_mode = mode
        else:
            raise ValueError(f"Unknown control mode: {mode!r}")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def setTargetTorques(self, target_torque) -> None:
        """Send a torque command to the 14 arm joints."""
        self.target_torque = target_torque
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.arm_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=self.target_torque,
        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def getJointStates(self) -> Tuple[List[float], List[float]]:
        """Return current arm joint positions (14) and velocities (14)."""
        states = p.getJointStates(self.robot, self.arm_joints)
        return [s[0] for s in states], [s[1] for s in states]

    def getLeftEEState(self) -> Tuple:
        """Return (position, orientation, linear_vel, angular_vel) of the left EE."""
        s = p.getLinkState(self.robot, self.left_ee, computeLinkVelocity=True)
        return s[0], s[1], s[6], s[7]

    def getRightEEState(self) -> Tuple:
        """Return (position, orientation, linear_vel, angular_vel) of the right EE."""
        s = p.getLinkState(self.robot, self.right_ee, computeLinkVelocity=True)
        return s[0], s[1], s[6], s[7]

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------

    def solveInverseDynamics(self, pos, vel, acc) -> List[float]:
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def getMassMatrix(self, joint_states):
        return p.calculateMassMatrix(self.robot, joint_states)

    def getJacobian(self, ee_link):
        """Return the linear Jacobian (3 × dof) for the given EE link."""
        joint_pos = [p.getJointState(self.robot, j)[0] for j in self.arm_joints]
        zeros = [0.0] * self.dof
        return p.calculateJacobian(
            self.robot, ee_link, [0, 0, 0], joint_pos, zeros, zeros)[0]

    def getJacobianOri(self, ee_link):
        """Return the angular Jacobian (3 × dof) for the given EE link."""
        joint_pos = [p.getJointState(self.robot, j)[0] for j in self.arm_joints]
        zeros = [0.0] * self.dof
        return p.calculateJacobian(
            self.robot, ee_link, [0, 0, 0], joint_pos, zeros, zeros)[1]
