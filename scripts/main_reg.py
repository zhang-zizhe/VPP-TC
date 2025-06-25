import argparse
import time
import pandas as pd
import torch
import numpy as np
import math
import os

import acc_functions
from safety_bounds import (
    gamma_model,
    online_search,
    online_gridsearch,
    online_search_external_reg,
    online_gridsearch_external_reg
)

from safety_bounds import gamma_external
from safety_bounds import gamma_external_reg
from bounds_zono import predict_zonotope
from external_transformer_reg import extransformer_reg
import sys

sys.path.append('./src')

import time
import numpy as np
import cvxpy as cp
import pybullet as p  # 新增：用于自碰撞检测


class Panda:
    def __init__(self, stepsize=1e-3, realtime=0):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque"
        self.position_control_gain_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.position_control_gain_d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.max_torque = [10000, 10000, 10000, 10000, 10000, 10000, 10000]

        self.cam_base_yaw = 30  # 初始 yaw
        self.cam_pitch = -20  # pitch 可以保持不变
        self.cam_dist = 1.0  # 距离可以保持不变
        self.cam_target = [0, 0, 0.5]  # 注视点

        # connect pybullet
        p.connect(p.GUI, options="--width=1280 --height=720")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=30, cameraPitch=-20,
                                     cameraTargetPosition=[0, 0, 0.5])

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        # p.setGravity(0,0,-10)
        p.setGravity(0, 0, 0)

        # load models
        # p.setAdditionalSearchPath("../models")

        self.plane = p.loadURDF("../models/urdf/plane/plane.urdf",
                                useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=.95)

        self.robot = p.loadURDF("../models/urdf/panda/panda.urdf",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)

        # # robot parameters
        self.dof = p.getNumJoints(self.robot) - 1  # Virtual fixed joint between the flange and last link

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []

        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j]) / 2.0)
            self.target_torque.append(0.)

        self.reset()

    def reset(self):
        self.t = 0.0
        self.control_mode = "torque"
        self.target_pos = [-1.669, -0.346, -0.842, -1.65, -0.367, 2.3, 1.99]
        for j in range(self.dof):
            # self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            self.target_torque[j] = 0.
            p.resetJointState(self.robot, j, targetValue=self.target_pos[j])

        self.resetController()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    # robot functions
    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(self.dof)])

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "velocity":
            self.control_mode = "velocity"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception('wrong control mode')

    def setTargetPositions(self, target_pos):
        self.target_pos = target_pos
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    # forces=self.max_torque,
                                    # positionGains=self.position_control_gain_p,
                                    # velocityGains=self.position_control_gain_d
                                    )

    def setTargetVelocity(self, target_vel):
        self.target_vel = target_vel
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetPositions=self.target_vel)

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.target_torque)

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel

    def solveInverseDynamics(self, pos, vel, acc):
        # print('start solve inverse dynamics')
        # print(list(p.calculateInverseDynamics(self.robot, pos, vel, acc)))
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot, 7, pos, ori))

    def solveForwardKinematics(self):
        pos = p.getLinkState(self.robot, 7)[0]
        ori = p.getLinkState(self.robot, 7)[1]
        return pos, ori

    def getEndVelocity(self):
        velocity = p.getLinkState(self.robot, 7, True)[6]
        return velocity

    def getEndAngularVelocity(self):
        velocity = p.getLinkState(self.robot, 7, True)[7]
        return velocity

    def applyForce(self, force):
        p.applyExternalForce(self.robot, 7, force, p.getLinkState(self.robot, 7)[0], p.WORLD_FRAME)

    def getJacobian(self):
        # velocity = p.getLinkState(self.robot, 7, True)[6]
        # ori = p.getLinkState(self.robot, 7)[1]
        # print(p.getJointState(self.robot, 0))
        jointpos = [0, 0, 0, 0, 0, 0, 0]
        jointpos[0] = p.getJointState(self.robot, 0)[0]
        jointpos[1] = p.getJointState(self.robot, 1)[0]
        jointpos[2] = p.getJointState(self.robot, 2)[0]
        jointpos[3] = p.getJointState(self.robot, 3)[0]
        jointpos[4] = p.getJointState(self.robot, 4)[0]
        jointpos[5] = p.getJointState(self.robot, 5)[0]
        jointpos[6] = p.getJointState(self.robot, 6)[0]
        jacobian = \
        p.calculateJacobian(self.robot, 7, [0, 0, 0], jointpos, [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])[0]
        return jacobian

    def getJacobian_ori(self):
        # velocity = p.getLinkState(self.robot, 7, True)[6]
        # ori = p.getLinkState(self.robot, 7)[1]
        # print(p.getJointState(self.robot, 0))
        jointpos = [0, 0, 0, 0, 0, 0, 0]
        jointpos[0] = p.getJointState(self.robot, 0)[0]
        jointpos[1] = p.getJointState(self.robot, 1)[0]
        jointpos[2] = p.getJointState(self.robot, 2)[0]
        jointpos[3] = p.getJointState(self.robot, 3)[0]
        jointpos[4] = p.getJointState(self.robot, 4)[0]
        jointpos[5] = p.getJointState(self.robot, 5)[0]
        jointpos[6] = p.getJointState(self.robot, 6)[0]
        jacobian = \
        p.calculateJacobian(self.robot, 7, [0, 0, 0], jointpos, [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])[1]
        return jacobian

    def getJacobianarbitary(self, jointstate):
        # velocity = p.getLinkState(self.robot, 7, True)[6]
        # ori = p.getLinkState(self.robot, 7)[1]
        # print(p.getJointState(self.robot, 0))
        jointpos = [0, 0, 0, 0, 0, 0, 0]
        jointpos[0] = p.getJointState(self.robot, 0)[0]
        jointpos[1] = p.getJointState(self.robot, 1)[0]
        jointpos[2] = p.getJointState(self.robot, 2)[0]
        jointpos[3] = p.getJointState(self.robot, 3)[0]
        jointpos[4] = p.getJointState(self.robot, 4)[0]
        jointpos[5] = p.getJointState(self.robot, 5)[0]
        jointpos[6] = p.getJointState(self.robot, 6)[0]
        jacobian = \
        p.calculateJacobian(self.robot, 7, [0, 0, 0], jointstate, [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])[0]
        return jacobian

    def getMassMatrix(self, jointstates):
        mass = p.calculateMassMatrix(self.robot, jointstates)
        return mass

    def getBallStates(self):
        ball_states = p.getBasePositionAndOrientation(self.ball)
        ball_pos = list(ball_states[0])
        ball_ori = list(ball_states[1])
        return ball_pos, ball_ori

    def getClosestPoints(self, link1, link2):
        mind = p.getClosestPoints(self.robot, self.robot, 1000, link1, link2)
        return mind


def get_args():
    parser = argparse.ArgumentParser(
        description="Estimate safe joint position bounds for a set of (q, qd) samples"
    )
    parser.add_argument(
        "--constraints",
        type=str,
        choices=["C1", "both", "none"],
        default="both",
        help="'C1' = only kinematic constraints (joint position and velocity); "
             "'both' = both C1 and self collision avoidance constraints; "
             "'none' = no constraint",

    )
    parser.add_argument(
        "--search",
        type=str,
        choices=["bisection", "grid"],
        default="bisection",
        help="'bisection' = online_search (parallel bisection); "
             "'grid' = online_gridsearch (fixed step scan)",
    )
    return parser.parse_args()


# ── 1.  添加球障碍物的工具函数 ──────────────────────────────
def create_obstacle_sphere(center, radius=0.1, rgba=(1, 0, 0, 1)):
    """
    在给定中心位置生成一个静态球形障碍物
    :param center: 圆心 [x, y, z] (m)
    :param radius: 半径 (m)，默认 5 cm
    :param rgba:   颜色 (r,g,b,a)
    :return: body_id
    """
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    body_id = p.createMultiBody(
        baseMass=0,  # 质量 0 → 静态
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=center
    )
    return body_id


def compute_bounds_external(q, qd, method, a_max_np,
                            *,
                            delta_t: float = 0.02,
                            tol: float = 1e-5,  # 仅 bisection 用
                            step_size: float = 0.1,  # 仅 grid 用
                            gamma_thresh: float = 0.2
                            ):
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=torch.float32)
    if not torch.is_tensor(qd):
        qd = torch.tensor(qd, dtype=torch.float32)

    if method == "bisection":
        return online_search_external_reg(
            q, qd,
            a_max_np=a_max_np,
            delta_t=delta_t,
            tol=tol,
            gamma_thresh=gamma_thresh
        )

    elif method == "grid":
        return online_gridsearch_external_reg(
            q, qd,
            a_max_np=a_max_np,
            delta_t=delta_t,
            step_size=step_size,
            gamma_thresh=gamma_thresh
        )

    else:
        raise ValueError(f"Unknown method: {method}")

def compute_bounds(q, qd, method, a_max_np):
    if method == "bisection":
        return online_search(
            q, qd, gamma_model, a_max_np,
            delta_t=0.02,
            tol=1e-5,
            threshold=2.55,
        )
    elif method == "grid":
        return online_gridsearch(
            q, qd, gamma_model, a_max_np,
            delta_t=0.02,
            step_size=0.1,
            threshold=2.53,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # gamma_safe = 0.15  # 满足 d_pred >= 0.14 m 视为“安全”
    # gamma_trig = 6*gamma_safe / 7  # 触发回溯阈值 (0.07 m)

    q_safe_prev = np.zeros(7)  # 最近安全姿态
    safe_prev_set = False  # 是否已经有记录
    retreating = False

    device_ext = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    times = []
    dists = []
    gammas = []

    # 确保输出目录存在
    os.makedirs("../output", exist_ok=True)

    args = get_args()

    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)

    duration = 50
    stepsize = 1e-3

    q_min_hardware = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max_hardware = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    robot = Panda(stepsize)
    robot.setControlMode("torque")

    lambda1 = 5
    lambda2 = 100
    lambda3 = 100

    obstacle_id = create_obstacle_sphere(
        center=[0.25, -0.25, 0.3],
        radius=0.1
    )

    Tau_SCA = []
    mindvector = []
    time.sleep(2)
    start_time = time.time()
    min_obs_dist = math.inf
    for link_id in range(robot.dof):
        cp_obs = p.getClosestPoints(robot.robot, obstacle_id,
                                    distance=1000.0, linkIndexA=link_id)
        if cp_obs:
            d_obs = min(cp[8] for cp in cp_obs)
            if d_obs < min_obs_dist:
                min_obs_dist = d_obs
        else:
            continue
    for i in range(int(duration / stepsize)):


        if i % int(1.0 / stepsize) == 0:
            print(f"Simulation time: {robot.t:.3f} s")

        end_pos = robot.solveForwardKinematics()[0]
        fx = -50 * (end_pos - np.array([0.35, -0.25, 0.3]))

        e1 = fx / np.linalg.norm(fx)
        e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        Lambda = np.diag([lambda1, lambda2, lambda3])
        D = Q @ Lambda @ Q.T

        xdot = robot.getEndVelocity()
        fc = -D @ (xdot - fx)

        q, qd = robot.getJointStates()

        q_t = torch.tensor(q, dtype=torch.float32, device=device_ext).unsqueeze(0)
        qd_t = torch.tensor(qd, dtype=torch.float32, device=device_ext).unsqueeze(0)

        gamma_obs_t = gamma_external_reg(q_t, qd_t)
        gamma_obs = gamma_obs_t.item()


        min_obs_dist = np.min([
            cp[8] for cp in p.getClosestPoints(robot.robot, obstacle_id, 1000)
        ])
        print(
            f"t={robot.t:.3f}s | "
            f"obs-dist={min_obs_dist:.5f}  γ_obs={gamma_obs:.3f}"
        )
        # if gamma_obs >= gamma_safe:
        #     q_safe_prev = np.array(q, copy=True)
        #     safe_prev_set = True
        #     retreating = False
        # elif gamma_obs < gamma_trig and safe_prev_set:
        #     retreating = True
        # # print(f"q: {np.round(np.array(q), 4)}")
        # # print(f"qd: {np.round(np.array(qd), 4)}")
        # if min_obs_dist < 0:
        #     print(f"External collision detected! Distance = {min_obs_dist:.5f}")
        #     print(f"Terminating at time {robot.t:.3f}s due to external collision.")
        #     exit(0)

        try:
            q_min_sca, q_max_sca = compute_bounds(torch.tensor(q), torch.tensor(qd), args.search, acc_max)
            # q_min, q_max = compute_bounds_zono(q, qd, Gamma, acc_max)
            q_min_e, q_max_e = compute_bounds_external(
                torch.tensor(q, dtype=torch.float32, device=device_ext),
                torch.tensor(qd, dtype=torch.float32, device=device_ext),
                method=args.search,
                a_max_np=acc_max,
            )


            print(f"t={robot.t:.3f}s | EE pos = {np.round(end_pos, 4)}")


        except Exception as e:
            print(f" skipped - {e}")
            continue

        # print('q', q)
        # print('qd', qd)
        # print('q_min', q_min)
        # print('q_max', q_max)

        # C1: Kinematic Constraints
        # qdd_lb, qdd_ub = acc_functions.compute_joint_acceleration_bounds_vec(q, qd,
        #                                                                      q_min_hardware, q_max_hardware,
        #                                                                      qd_lim, acc_max,
        #                                                                      dt=0.02, viability=True)

        # C2: SCA Constraints
        qdd_lb2, qdd_ub2 = acc_functions.compute_joint_acceleration_bounds_vec(q, qd,
                                                                               q_min_e, q_max_e,
                                                                               qd_lim, acc_max,
                                                                               dt=0.02, viability=False)
        qdd_lb, qdd_ub = qdd_lb2, qdd_ub2

        # Last compute torque bounds
        # tau_lb = robot.solveInverseDynamics(q, q_dot, list(qdd_lb))
        # tau_ub = robot.solveInverseDynamics(q, q_dot, list(qdd_ub))

        tau_id = robot.solveInverseDynamics(q, qd, [0] * 7)
        b = np.array(robot.getMassMatrix(q)) @ np.zeros(7) - tau_id
        # tau_lb = robot.getMassMatrix(q) @ qdd_lb - b
        # tau_ub = robot.getMassMatrix(q) @ qdd_ub - b
        # tau_lb[6] = 0
        # tau_ub[6] = 0
        # print('tau_lb:', tau_lb)
        # print('tau_ub:', tau_ub)

        x = cp.Variable(7)  # Torque
        y = cp.Variable(7)  # q̈

        J = np.array(robot.getJacobian())  # 3×7 的雅可比矩阵
        JT_pinv = np.linalg.pinv(J.T)  # (J^T)^+，shape 3×7

        if args.constraints == "C1":
            for i in range(7):
                if qdd_lb[i] - qdd_ub[i] > -0.001:
                    qdd_lb[i] = qdd_ub[i] - 0.0001

        # print('qdd_lb:', qdd_lb2)
        # print('qdd_ub:', qdd_ub2)
        # print('qdd_lb2:', qdd_lb2)
        # print('qdd_ub2:', qdd_ub2)

        qdd_ub_opt = np.minimum(qdd_ub, qdd_ub2)
        qdd_lb_opt = np.maximum(qdd_lb, qdd_lb2)

        for i in range(7):
            if qdd_lb_opt[i] - qdd_ub_opt[i] > -0.001:
                qdd_lb_opt[i] = qdd_ub_opt[i] - 0.0001

        if retreating:
            dir_sign = np.sign(q_safe_prev - np.array(q))
            qdd_cmd = np.where(dir_sign > 0, qdd_ub_opt,
                               np.where(dir_sign < 0, qdd_lb_opt, 0.0))

            tau_retreat = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
            tau_max = np.array(robot.max_torque)
            tau_retreat = np.clip(tau_retreat, -tau_max, tau_max)  # 饱和

            robot.setTargetTorques(tau_retreat)
            robot.step()
            times.append(robot.t)
            time.sleep(robot.stepsize)
            continue

        if args.constraints == "C1":
            prob = cp.Problem(
                cp.Minimize(cp.sum_squares(JT_pinv @ x - fc)),
                [np.array(robot.getMassMatrix(q)) @ y - x == b,
                 y >= qdd_lb,
                 y <= qdd_ub],
            )
        elif args.constraints == "both":
            prob = cp.Problem(
                cp.Minimize(cp.sum_squares(JT_pinv @ x - fc)),
                [np.array(robot.getMassMatrix(q)) @ y - x == b,
                 y >= qdd_lb_opt,
                 y <= qdd_ub_opt],
            )
        elif args.constraints == "none":
            prob = cp.Problem(
                cp.Minimize(cp.sum_squares(JT_pinv @ x - fc)),
                [np.array(robot.getMassMatrix(q)) @ y - x == b],
            )
        else:
            print("Wrong constraint input.")

        prob.solve(solver=cp.OSQP)

        target_torque = x.value.tolist()

        robot.setTargetTorques(target_torque)

        robot.step()
        times.append(robot.t)
        # dists.append(dist)

        time.sleep(robot.stepsize)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total runtime: {elapsed:.2f} seconds")
    df = pd.DataFrame({
        "time": times,
        "dist": dists,
        "gamma": gammas
    })
    output_path = f"../output/{args.constraints}.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")