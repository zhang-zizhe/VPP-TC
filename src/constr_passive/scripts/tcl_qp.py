#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import ctypes
import os
import rospkg
import functions

class subscriber_QP:
    def __init__(self):
        self.sub_sca = rospy.Subscriber("sca_params", Float32MultiArray, self.QP_callback_sub_sca, queue_size=10)
        self.sub_joint = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=10)
        self.sub_ee = rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.QP_callback_sub_ee, queue_size=10)
        self.sub_state = rospy.Subscriber("/franka_state_controller/franka_model", Float32MultiArray, self.QP_callback_sub_state, queue_size=10)

        self.message_sca = [0, 0, 0, 0, 0, 0, 0, 0]
        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_state = list(np.zeros(105))

    def QP_callback_sub_sca(self, msg):
        self.message_sca = msg.data

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position = msg.position
        self.message_joint_velocity = msg.velocity

    def QP_callback_sub_ee(self, msg):
        self.message_ee = [msg.position.x, msg.position.y, msg.position.z]
 
    def QP_callback_sub_state(self, msg):
        self.message_state = msg.data
    
    def return_message(self):
        return self.message_sca, self.message_joint_position, self.message_joint_velocity, self.message_ee, self.message_state


def CVXsolver_in_python(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in):
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('constr_passive')   
    so_path = os.path.join(pkg_path, 'scripts', 'libcvxgensolver_joint_lim.so')

    my_c_library = ctypes.CDLL(so_path)
    my_c_library.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double)]
    my_c_library.CVXsolver.restype = None

    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    c_b_in = (ctypes.c_double * len(b_in))(*b_in)
    c_qdd_lb_in = (ctypes.c_double * len(qdd_lb_in))(*qdd_lb_in)
    c_qdd_ub_in = (ctypes.c_double * len(qdd_ub_in))(*qdd_ub_in)
    
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()

    my_c_library.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in, c_qdd_lb_in, c_qdd_ub_in,
                           ctypes.byref(opt_x1), ctypes.byref(opt_x2),
                           ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))

    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x


if __name__ == "__main__":

    rospy.init_node("QP")
    rospy.logwarn("Node initialized!")
    r = rospy.Rate(200)
    
    
    Subscriber_QP = subscriber_QP()

  
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size = 10)


    lambda1, lambda2, lambda3 = 8, 48, 48
    alpha = 1e-4 
    

    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    
 
    target_pos = np.array([0.0, -0.6, 0.3])
    dt = 0.02

    while not rospy.is_shutdown():
        
        try:
            q = Subscriber_QP.return_message()[1]
            q_dot = Subscriber_QP.return_message()[2]
            end_pos = Subscriber_QP.return_message()[3]
        
            
            Jacobian_raw = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
            Jacobian = Jacobian_raw[0:3, :]  
            
            
            if np.linalg.cond(Jacobian) > 1e10:
                rospy.logwarn("Jacobian matrix is ill-conditioned, skipping iteration")
                r.sleep()
                continue

            
            xdot = Jacobian @ np.array(q_dot)

            MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
            Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
            Gravity = np.array(Subscriber_QP.return_message()[4][7:14])

            fx = -50 * (np.array(end_pos) - target_pos)
            e1 = fx / np.linalg.norm(fx)
            e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
            e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1, e2)
            Q = np.column_stack((e1, e2, e3))
            Lambda = np.diag([lambda1, lambda2, lambda3])
            D = Q @ Lambda @ Q.T
            fc = -D @ (xdot - fx)
            
            qdd_lb, qdd_ub = functions.compute_joint_acceleration_bounds_vec(
                q, q_dot, q_min, q_max, qd_lim, acc_max, dt, viability=True
            )
            
            
            try:
                M_inv = np.linalg.inv(MassMatrix)
            except np.linalg.LinAlgError:
                rospy.logwarn("Mass matrix is singular, using pseudo-inverse")
                M_inv = np.linalg.pinv(MassMatrix)
            
            qdd_zero = np.zeros(7)
            tau_id = Coriolis + Gravity
            
            for idx in range(7):
                if qdd_lb[idx] > qdd_ub[idx]:
                    qdd_lb[idx] = qdd_ub[idx] - 1e-4
            
            target_torque, success = solve_qp_with_cvxpy(
                Jacobian, fc, MassMatrix, tau_id, qdd_lb, qdd_ub, alpha
            )
            
            if not success or target_torque is None:
                rospy.logwarn("QP solver failed, using fallback control")
                target_torque = Jacobian.T @ fc
            
            target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
            
            msg_torque = Float32MultiArray()
            msg_torque.data = list(target_torque)
            pub.publish(msg_torque)
            
            rospy.logwarn(f"Loop iteration completed, target_torque: {target_torque[:3]}...")
            
        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            rospy.logerr(f"Error details: {type(e).__name__}")
