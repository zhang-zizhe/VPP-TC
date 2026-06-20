#!/usr/bin/env python3

import gc
import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import rospkg
import ctypes
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_qp as functions_bounds
import functions as functions_collision

class Settings(ctypes.Structure):
    _fields_ = [
        ("resid_tol", ctypes.c_double),
        ("eps", ctypes.c_double),
        ("max_iters", ctypes.c_int),
        ("refine_steps", ctypes.c_int),
        ("verbose", ctypes.c_int),
        ("debug", ctypes.c_int),
        ("kkt_reg", ctypes.c_double),
    ]
def CVXsolver_basic(J_in, fc_in, M_in, tau_in, qdd_lb_in, qdd_ub_in):
    my_c_library = ctypes.CDLL('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/basic.so')
    my_c_library.set_defaults()
    my_c_library.setup_indexing()
    my_c_library.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double)]
    # Get reference to the global variable
    # solver_settings = ctypes.cast(
    #     ctypes.addressof(my_c_library.settings), ctypes.POINTER(Settings)
    # )

    # # Modify its values
    # solver_settings.contents.verbose = 1
    # solver_settings.contents.debug = 1
    my_c_library.CVXsolver.restype = None
    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    # print("J_in:", list(c_J_in))
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    # print("fc_in:", list(c_fc_in))
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    # print("M_in:", list(c_M_in))
    c_b_in = (ctypes.c_double * len(tau_in))(*tau_in)
    # print("b_in:", list(c_b_in))
    c_qdd_lb_in = (ctypes.c_double * len(qdd_lb_in))(*qdd_lb_in)
    # print("qdd_lb_in:", list(c_qdd_lb_in))
    c_qdd_ub_in = (ctypes.c_double * len(qdd_ub_in))(*qdd_ub_in)
    # print("qdd_ub_in:", list(c_qdd_ub_in))
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()
    my_c_library.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in,c_qdd_lb_in,  c_qdd_ub_in,
                           ctypes.byref(opt_x1), ctypes.byref(opt_x2),
                           ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))
    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x

def CVXsolver_sca(J_in, fc_in, M_in, tau_in, qdd_lb_in, qdd_ub_in, g_in, e_in, c_in):
    my_c_library = ctypes.CDLL('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/sca.so')
    my_c_library.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    my_c_library.CVXsolver.restype = None
    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    # print("J_in:", c_J_in)
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    # print("fc_in:", c_fc_in)
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    # print("M_in:", c_M_in)
    c_b_in = (ctypes.c_double * len(tau_in))(*tau_in)
    # print("b_in:", c_b_in)
    c_qdd_lb_in = (ctypes.c_double * len(qdd_lb_in))(*qdd_lb_in)
    # print("qdd_lb_in:", c_qdd_lb_in)
    c_qdd_ub_in = (ctypes.c_double * len(qdd_ub_in))(*qdd_ub_in)
    # print("qdd_ub_in:", c_qdd_ub_in)
    c_g_in = (ctypes.c_double * len(g_in))(*g_in)
    c_e_in = (ctypes.c_double * len(e_in))(*e_in)
    c_c_in = (ctypes.c_double * len(c_in))(*c_in)
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()
    my_c_library.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in, c_qdd_lb_in, c_qdd_ub_in, c_g_in, c_e_in, c_c_in,
                           ctypes.byref(opt_x1), ctypes.byref(opt_x2),
                           ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))
    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x

class subscriber_QP:
    def __init__(self):
        self.sub_joint = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=1)
        self.sub_ee = rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.QP_callback_sub_ee, queue_size=1)
        self.sub_state = rospy.Subscriber("/franka_state_controller/franka_model", Float32MultiArray, self.QP_callback_sub_state, queue_size=1)

        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_state = list(np.zeros(105))

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
    r = rospy.Rate(1500)  
    # gc.disable()
    
    Subscriber_QP = subscriber_QP()
    
    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size=1)

    lambda1, lambda2, lambda3 = 9, 10, 10
    alpha = 1e-3
    
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    # q_min = np.array([-2.5, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8])
    # q_max = np.array([ 2.5,  1.7,  2.8, -0.1,  2.8,  3.7,  2.8])
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])*0.5
    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    
    target_pos = np.array([0, -0.3, 0.4])
    dt = 0.02  
    rospy.sleep(1.0)
    t0 = time.time()
    while not rospy.is_shutdown():
        # gc.enable()
        
        # if time.time() - t0 > 100:
        #     gc.collect()
        #     t0 = time.time()

        q = Subscriber_QP.return_message()[1]
        q_dot = Subscriber_QP.return_message()[2]
        end_pos = Subscriber_QP.return_message()[3]
    
        
        Jacobian_raw = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
        Jacobian = Jacobian_raw[0:3, :].T 
        # print("Jacobian:", Jacobian)
        
        xdot = Jacobian.T @ np.array(q_dot)  

        MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
        Gravity = np.array(Subscriber_QP.return_message()[4][7:14])

        fx = -2* (np.array(end_pos) - target_pos)
        if np.linalg.norm(np.array(end_pos) - target_pos) < 0.2:
            fx = -max(2.5*(0.2/np.linalg.norm(np.array(end_pos) - target_pos)),8)* (np.array(end_pos) - target_pos)
        e1 = fx / np.linalg.norm(fx)
        e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        Lambda = np.diag([lambda1, lambda2, lambda3])
        D = Q @ Lambda @ Q.T
        fc = -D @ (xdot - fx)
        
        start = time.time()
        qdd_lb, qdd_ub, flags = functions_bounds.compute_joint_acceleration_bounds_vec(
            q, q_dot, q_min, q_max, qd_lim, acc_max, dt*0.25, viability=True
        )
        # print("flags:", flags)
        
        if flags.any() != 0:
            # rospy.logwarn("Joint limits are active!")
            indices = np.nonzero(flags)[0]
            # rospy.logwarn("Active joint limits indices: %s", indices)
            for idx in indices:
                if flags[idx] == 1:
                    rospy.logwarn("Joint %d position limit is active.", idx)
                elif flags[idx] == 2:
                    rospy.logwarn("Joint %d velocity limit is active.", idx)
                elif flags[idx] == 3:
                    rospy.logwarn("Joint %d viability limit is active.", idx)
            # rospy.logwarn("Flags values: %s", flags[indices])


        M_inv = np.linalg.pinv(MassMatrix)
        qdd_zero = np.zeros(7)
        tau_id = MassMatrix @ qdd_zero + Coriolis #+ Gravity 

        # # --- Self-collision gamma and gradient ---
        # start = time.perf_counter()
        gamma_val, grad_gamma = functions_collision.compute_gamma_and_grad(
            np.array(q, dtype=np.float32), np.array(q_dot, dtype=np.float32), threshold=13
        )
        print("gamma:", gamma_val)

        # gamma_val = functions_collision.compute_gamma(
        #     np.array(q, dtype=np.float32), np.array(q_dot, dtype=np.float32)
        # )
        # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
        # print("grad:", grad_gamma)
        # if gamma_val < 10.1:
        #     pass
        # # gamma_val = 11
        # grad_gamma = None
        
        
        
        for idx in range(7):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4

        # print("qdd_lb:", qdd_lb)
        # print("qdd_ub:", qdd_ub)
        
        # u = cp.Variable(7)
        # JT_pinv = np.linalg.pinv(Jacobian)
        # objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)
        # constraints = [
        #     M_inv @ u >= qdd_lb + M_inv @ tau_id,
        #     M_inv @ u <= qdd_ub + M_inv @ tau_id,
        # ]
        J_in = list(np.linalg.pinv(Jacobian).flatten('F'))
        fc_in = list(fc)
        M_in = list(np.array(M_inv).flatten('F'))
        b_in = list(tau_id)
        qdd_lb_in = list(qdd_lb)
        qdd_ub_in = list(qdd_ub)
        # qdd_lb_in = list(-acc_max)
        # qdd_ub_in = list(acc_max)


        
        soft = False
        if grad_gamma is not None:
            rospy.logwarn("Self Collision Avoidance limits are active!")
            
            grad_q  = grad_gamma[:7]
            grad_qd = grad_gamma[7:]
            g_eff   = 0.5 * grad_q * dt**2 + grad_qd * dt
            g_eff_in = list(g_eff)
            c_const = float(np.dot(np.asarray(grad_q, dtype=np.float64), np.asarray(q_dot, dtype=np.float64)) * dt)
            c_in = [c_const]
            # eps     = ((1.305 - (gamma_val*0.1))**0.5)*0.5
            eps     = ((1.305 - (gamma_val*0.1))**0.8)*0.8
            e_in = [eps]
            try:
                target_torque = CVXsolver_sca(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in, list(g_eff), e_in, c_in)

            except Exception as e:
                rospy.logwarn("CVXsolver_sca failed")
                target_torque = list(Jacobian.T @ fc)
        else:
        # if True:
            try:
                # start = time.time()
                target_torque = CVXsolver_basic(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in)
                # print(target_torque)
                
                # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
                
            except Exception as e:
                rospy.logwarn("CVXsolver_basic failed")
                target_torque = list(Jacobian.T @ fc)
 
        end = time.time()
        # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
        if not soft:
            # target_torque = np.array(u.value)
            # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
            # target_torque = [-14, -14, -14, -14, -2, -2, -2]
            target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
            msg_torque = Float32MultiArray()
            msg_torque.data = target_torque
            # print("torque:", target_torque)
            pub.publish(msg_torque)
        
        
    
    
        # r.sleep()
