#!/usr/bin/env python3

import rospy
import numpy as np
import cvxpy as cp
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_qp as functions_bounds
import functions as functions_collision
import time
import gc

class subscriber_QP:
    def __init__(self):
        # self.sub_sca = rospy.Subscriber("sca_params", Float32MultiArray, self.QP_callback_sub_sca, queue_size=1)
        self.sub_joint = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=1)
        self.sub_ee = rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.QP_callback_sub_ee, queue_size=1)
        self.sub_state = rospy.Subscriber("/franka_state_controller/franka_model", Float32MultiArray, self.QP_callback_sub_state, queue_size=1)

        self.message_sca = [0, 0, 0, 0, 0, 0, 0, 0]
        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_state = list(np.zeros(105))

    # def QP_callback_sub_sca(self, msg):
    #     self.message_sca = msg.data

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position = msg.position
        self.message_joint_velocity = msg.velocity

    def QP_callback_sub_ee(self, msg):
        self.message_ee = [msg.position.x, msg.position.y, msg.position.z]
 
    def QP_callback_sub_state(self, msg):
        self.message_state = msg.data
    
    def return_message(self):
        return 1, self.message_joint_position, self.message_joint_velocity, self.message_ee, self.message_state
if __name__ == "__main__":
    rospy.init_node("cvxpy_QP")
    rospy.logwarn("CVXPY QP Node initialized!")
    r = rospy.Rate(1000)  
    gc.disable()
    
    Subscriber_QP = subscriber_QP()
    
    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals_test", Float32MultiArray, queue_size=1)

    lambda1, lambda2, lambda3 = 2, 10, 10
    alpha = 1e-3
    
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    
    target_pos = np.array([0, -0.3, 0.4])
    dt = 0.02  

    rospy.sleep(1.0)

    while not rospy.is_shutdown():
        time_start = time.time()
        # print("----------------------------------")
        
        q = Subscriber_QP.return_message()[1]
        q_dot = Subscriber_QP.return_message()[2]
        end_pos = Subscriber_QP.return_message()[3]
    
        
        Jacobian_raw = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
        Jacobian = Jacobian_raw[0:3, :].T 
        
        xdot = Jacobian.T @ np.array(q_dot)  

        MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
        Gravity = np.array(Subscriber_QP.return_message()[4][7:14])

        fx = -2* (np.array(end_pos) - target_pos)
        e1 = fx / np.linalg.norm(fx)
        e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        Lambda = np.diag([lambda1, lambda2, lambda3])
        D = Q @ Lambda @ Q.T
        fc = -D @ (xdot - fx)
        
        qdd_lb, qdd_ub = functions_bounds.compute_joint_acceleration_bounds_vec(
            q, q_dot, q_min, q_max, qd_lim, acc_max, dt, viability=True
        )
        
        time1 = time.time()
        # print("time before gamma:", time1 - time_start)
        
        M_inv = np.linalg.pinv(MassMatrix)           
        qdd_zero = np.zeros(7)
        tau_id = MassMatrix @ qdd_zero + Coriolis #+ Gravity 
        # print("time:", rospy.get_time())
        start = time.perf_counter()

        # --- Self-collision gamma and gradient ---
        gamma_val, grad_gamma = functions_collision.compute_gamma_and_grad(
            np.array(q, dtype=np.float32), np.array(q_dot, dtype=np.float32), threshold=7
        )
        print("gamma:", gamma_val)
        
        # gamma_val = functions_collision.compute_gamma(
        #     np.array(q, dtype=np.float32), np.array(q_dot, dtype=np.float32)
        # )
        end = time.perf_counter()
        print(f"Execution time for gamma and grad: {(end - start) * 1e3:.2f} ms")
        time2 = time.time()
        # print("time after gamma:", time2 - time_start)
        # gamma_val = 11
        # grad_gamma = None
        
        
        for idx in range(7):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4
        
        u = cp.Variable(7)
        JT_pinv = np.linalg.pinv(Jacobian)
        objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)
        constraints = [
            M_inv @ u >= qdd_lb + M_inv @ tau_id,
            M_inv @ u <= qdd_ub + M_inv @ tau_id,
        ]
        time3 = time.time()
        print("time before QP:", time3 - time_start)
        soft = False
        if grad_gamma is not None:
            
            grad_q  = grad_gamma[:7]
            grad_qd = grad_gamma[7:]
            g_eff   = 0.5 * grad_q * dt**2 + grad_qd * dt
            c_const = float(np.dot(np.asarray(grad_q, dtype=np.float64), np.asarray(q_dot, dtype=np.float64)) * dt)
            eps     = 2e-1
            constraints.append(g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id)
            # print("sca constraint added")
            prob = cp.Problem(cp.Minimize(objective), constraints)
            try:
                prob.solve(solver=cp.OSQP)
                # print("QP solved")
            except cp.SolverError:
                rospy.logwarn("QP infeasible, relax constraint, use soft constraint")
                soft = True
                qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                target_torque = MassMatrix @ qdd_cmd + Coriolis#+Gravity
                # target_torque = np.clip(target_torque, [-14, -14, -14, -14, -2, -2, -2], [14, 14, 14, 14, 2, 2, 2])
                msg_torque = Float32MultiArray()
                msg_torque.data = target_torque.tolist() 
                pub.publish(msg_torque)
                # r.sleep()
                continue

            while prob.status != cp.OPTIMAL and eps > 1e-3:
                rospy.logwarn("QP infeasible, relax constraint, reduce epsilon")
                eps = eps - 1e-1
                constraints[-1] = (g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id)
                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.OSQP)
            if eps < 1e-3:
            # if True:
                rospy.logwarn("QP still infeasible, relax constraint, use soft constraint")
                soft = True
                qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                target_torque = MassMatrix @ qdd_cmd + Coriolis#+Gravity
                # target_torque = np.clip(target_torque, [-14, -14, -14, -14, -2, -2, -2], [14, 14, 14, 14, 2, 2, 2])
                msg_torque = Float32MultiArray()
                msg_torque.data = target_torque.tolist() 
                pub.publish(msg_torque)
                # r.sleep()
                continue
        else:
        # if True:
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP)
            time4 = time.time()
            # print("time after QP:", time4 - time_start)
            # print("test")

        if not soft:
            target_torque = np.array(u.value)
            # target_torque = np.array([-14, -14, -14, -14, -2, -2, -2])
            # target_torque = np.clip(target_torque, [-14, -14, -14, -14, -2, -2, -2], [14, 14, 14, 14, 2, 2, 2])
            msg_torque = Float32MultiArray()
            msg_torque.data = target_torque.tolist()
            pub.publish(msg_torque)
        
        
    
    
    r.sleep()

