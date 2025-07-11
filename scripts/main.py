# main.py
import time
import pandas as pd
import numpy as np
import math
import os

import functions

import sys
sys.path.append('./src')

import time
import numpy as np
import cvxpy as cp
import pybullet as p           # 新增：用于自碰撞检测
from Panda import Panda

def feasible_qdd_region(
    grad_gamma: np.ndarray,  # shape (14,)
    qd: np.ndarray,           # shape (7,)
    dt: float,
    qdd_lb: np.ndarray,       # shape (7,)
    qdd_ub: np.ndarray        # shape (7,)
):
    """
    返回 Delta gamma 在盒约束 [qdd_lb, qdd_ub] 上的最小值和最大值，
    以及判断是否存在能使 Delta gamma > 0 的 qdd。
    
    Δγ ≈ q̈ᵀ * g_eff, 其中
      g_eff = 0.5 * ∇_qγ * dt**2 + ∇_{q̇}γ * dt.
    """
    # 拆分梯度
    grad_q  = grad_gamma[:7]
    grad_qd = grad_gamma[7:]

    # 1) 计算有效梯度方向
    g_eff = 0.5 * grad_q * dt**2 + grad_qd * dt   # shape (7,)
    c = grad_q.dot(qd) * dt

    # 2) 在盒约束上，Δγ 的最小/最大值出现在每个维度要么取下界，要么取上界
    #    如果 g_eff[i] > 0，则 Δγ 对 q̈[i] 单调递增 → 最小在下界，最大在上界
    #    如果 g_eff[i] < 0，则 Δγ 对 q̈[i] 单调递减 → 最小在上界，最大在下界
    corner_min = np.where(g_eff>0, qdd_lb, qdd_ub)
    corner_max = np.where(g_eff>0, qdd_ub, qdd_lb)

    delta_gamma_min = np.dot(g_eff, corner_min)+ c
    delta_gamma_max = np.dot(g_eff, corner_max)+ c

    # 3) 可行性判断
    exists_positive = (delta_gamma_max > 0) and (delta_gamma_min < delta_gamma_max)
    if not exists_positive:
        print("Warning: no feasible qdd in original box can increase Gamma!")
        return corner_max

    # new_lb = qdd_lb.copy()
    # new_ub = qdd_ub.copy()
    # # 计算用于“最坏情况”迭代的 delta_min
    # # （此时 corner_min 对所有 j 都是 worst-case for Δγ）
    # for i in range(7):
    #     # 其他分量固定在 corner_min 上
    #     # worst_other = c + Σ_{j≠i} g_eff[j]*corner_min[j]
    #     worst_other = delta_gamma_min - g_eff[i] * corner_min[i]

    #     if g_eff[i] > 0:
    #         # 需要 g_eff[i]*q̈[i] > - worst_other
    #         bound_i = - worst_other / g_eff[i]
    #         new_lb[i] = max(new_lb[i], bound_i)
    #     else:
    #         # g_eff[i] < 0 时只收缩上界
    #         bound_i = - worst_other / g_eff[i]
    #         new_ub[i] = min(new_ub[i], bound_i)
    return corner_max

if __name__ == "__main__":
    times, dists, gammas = [], [], []
    os.makedirs("../output", exist_ok=True)

    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
    qd_lim  = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)
    q_min_hardware = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max_hardware = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    duration, stepsize = 15.0, 1e-3
    robot = Panda(stepsize)
    robot.setControlMode("torque")

    lambda1, lambda2, lambda3 = 5, 100, 100
    alpha = 1e-2  # 阻尼项权重

    time.sleep(2)
    start_time = time.time()

    for i in range(int(duration / stepsize)):
        if i % int(1.0 / stepsize) == 0:
            print(f"Simulation time: {robot.t:.3f} s")

        # --- 计算主任务的末端力 fc ---
        end_pos = robot.solveForwardKinematics()[0]
        fx = -50 * (end_pos - np.array([0, 0, 0.3]))
        e1 = fx / np.linalg.norm(fx)
        e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        Lambda = np.diag([lambda1, lambda2, lambda3])
        D = Q @ Lambda @ Q.T
        xdot = robot.getEndVelocity()
        fc = -D @ (xdot - fx)

        # --- 读取关节状态 & 自碰撞 gamma & 梯度 ---
        q, qd = robot.getJointStates()
        Gamma, grad_gamma = functions.compute_gamma_and_grad(q, qd, threshold=2.5)

        qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
            q, qd, q_min_hardware, q_max_hardware, qd_lim, acc_max, dt=0.02, viability=True)

        # 如果需要紧急避碰，直接用梯度方向
        if grad_gamma is not None:
            corner = feasible_qdd_region(
                grad_gamma, qd, dt=0.02, qdd_lb=qdd_lb, qdd_ub=qdd_ub)
            # print(feasible_region)
            # k = 1
            # # print(f"t={robot.t:.3f}s: grad_gamma={grad_gamma}, Gamma={Gamma}")
            # ddq_cmd = k * grad_gamma[7:14]
            ddq_cmd = corner
            # ddq_cmd[0] = 0.0
            ddq_cmd = np.clip(ddq_cmd, -acc_max, acc_max)
            tau_cmd = robot.solveInverseDynamics(q, qd, ddq_cmd.tolist())
            robot.setTargetTorques(tau_cmd)
            robot.step()
        
        else:

            for idx in range(7):
                if qdd_lb[idx] > qdd_ub[idx]:
                    qdd_lb[idx] = qdd_ub[idx] - 1e-4

            # --- 构建 QP：最小化 ∥J⁺ x - fc∥² + α ∥y∥² ---
            M = np.array(robot.getMassMatrix(q))
            tau_id = robot.solveInverseDynamics(q, qd, [0]*7)
            b = M @ np.zeros(7) - tau_id

            x = cp.Variable(7)  # torque
            y = cp.Variable(7)  # acceleration
            J = np.array(robot.getJacobian())
            JT_pinv = np.linalg.pinv(J.T)

            objective = cp.sum_squares(JT_pinv @ x - fc) + alpha * cp.sum_squares(y)

            constraints = [
                M @ y - x == b,
                y >= qdd_lb,
                y <= qdd_ub,
            ]

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP)

            # --- 执行控制 & 可视化 ---
            robot.setTargetTorques(x.value.tolist())
            robot.step()

        # 更新相机视角（可选）
        # new_yaw = (robot.cam_base_yaw - 60.0 * robot.t) % 360
        # p.resetDebugVisualizerCamera(
        #     cameraDistance=robot.cam_dist,
        #     cameraYaw=new_yaw,
        #     cameraPitch=robot.cam_pitch,
        #     cameraTargetPosition=robot.cam_target
        # )

        # --- 自碰撞检测 & 日志 ---
        dist, collision = math.inf, False
        for l1 in range(7):
            for l2 in range(7):
                if abs(l1 - l2) > 1 and not {l1, l2} == {4,6}:
                    pts = robot.getClosestPoints(l1, l2)
                    dmin = min(cp[8] for cp in pts)
                    dist = min(dist, dmin)
                    if dmin < 0:
                        collision = True
        times.append(robot.t)
        dists.append(dist)
        gammas.append(Gamma)

        if collision:
            print(f"t={robot.t:.3f}s: Collision! dist={dist}")
            break
        else:
            print(f"t={robot.t:.3f}s: Safe. dist={dist}, gamma={Gamma}")
            pass

        time.sleep(robot.stepsize)

    # --- 保存结果 ---
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.2f}s")
    df = pd.DataFrame({"time": times, "dist": dists, "gamma": gammas})
    df.to_csv(f"../output/{time.time()}.csv", index=False)
    print("Results saved.")