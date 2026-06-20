#!/usr/bin/env python3

import gc
import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
import rospkg
import ctypes
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_qp as functions_bounds
import functions as functions_collision
from rdf import query_sdf, query_sdf_batch
import tf.transformations as tft



import rospy, tf2_ros
from geometry_msgs.msg import TransformStamped

def to_vector_or_none(x, expect_len_set=(7, 14)):
    """把任意 x 转为一维 ndarray；若为 None/空/含NaN/Inf/长度不在期望集合，则返回 None。"""
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if expect_len_set is not None and arr.size not in expect_len_set:
        return None
    return arr
def publish_world_alias():
    br = tf2_ros.StaticTransformBroadcaster()
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"          # parent
    t.child_frame_id = "panda_link0"  # child
    t.transform.translation.x = 0.0
    t.transform.translation.y = 0.0
    t.transform.translation.z = 0.0
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = 1.0
    br.sendTransform(t)



from visualization_msgs.msg import Marker
def make_sphere_marker(position, scale=0.05,
                       color=(0.1, 0.7, 1.0), alpha=0.95,
                       mid=0, ns="targets"):
    """
    Always create a sphere marker in the 'world' frame.

    Args:
        position: (x, y, z)
        scale: sphere diameter in meters
        color: (r, g, b) in [0, 1]
        alpha: transparency in [0, 1]
        mid: marker id
        ns: namespace string
    """
    m = Marker()
    m.header.frame_id = "panda_link0_sc"     # <- fixed to world
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(position[0])
    m.pose.position.y = float(position[1])
    m.pose.position.z = float(position[2])
    m.pose.orientation.w = 1.0
    m.scale.x = scale
    m.scale.y = scale
    m.scale.z = scale
    m.color.r, m.color.g, m.color.b = color
    m.color.a = alpha
    m.lifetime = rospy.Duration(0)   # 0 = forever
    m.frame_locked = True
    return m


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
        # self.sub_obs1 = rospy.Subscriber("/natnet_ros/obs1/pose", PoseStamped, self.QP_callback_sub_obs1, queue_size=10)
        # self.sub_obs2 = rospy.Subscriber("/natnet_ros/obs2/pose", PoseStamped, self.QP_callback_sub_obs2, queue_size=10)
        self.sub_sca = rospy.Subscriber("/sca/value",
                                        Float32, self.QP_callback_sub_sca, queue_size=1)
        self.sub_eca = rospy.Subscriber("/eca/value",
                                        Float32, self.QP_callback_sub_eca, queue_size=1)
        self.sub_sca_grd = rospy.Subscriber("/sca/grad",
                                        Float32MultiArray, self.QP_callback_sub_sca_grd, queue_size=1)
        self.sub_eca_grd = rospy.Subscriber("/eca/grad",
                                        Float32MultiArray, self.QP_callback_sub_eca_grd, queue_size=1)
        
        self.oculus_pos = rospy.Subscriber('/oculus/desired_pose', PoseStamped, self.oculus_pos_callback, queue_size=1)

        self.message_sca = None
        self.message_eca = None
        self.sca_ready = False
        self.eca_ready = False
        self.message_sca_grd = None
        self.message_eca_grd = None
        self.sca_grd_ready = False
        self.eca_grd_ready = False
        self.message_oculus = None

        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_obs1 = [0, 0, 0]
        self.message_obs2 = [0, 0, 0]
        self.message_state = list(np.zeros(105))

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position = msg.position
        self.message_joint_velocity = msg.velocity

    def QP_callback_sub_ee(self, msg):
        self.message_ee = [msg.position.x, msg.position.y, msg.position.z]
        self.message_ee_quat = [msg.orientation.x, msg.orientation.y,
                                msg.orientation.z, msg.orientation.w]

    # def QP_callback_sub_obs1(self, msg):
    #     # PoseStamped 里位置在 msg.pose.position
    #     self.message_obs1 = [msg.pose.position.x,
    #                          msg.pose.position.y,
    #                          msg.pose.position.z]

    # def QP_callback_sub_obs2(self, msg):
    #     # PoseStamped 里位置在 msg.pose.position
    #     self.message_obs2 = [msg.pose.position.x,
    #                          msg.pose.position.y,
    #                          msg.pose.position.z]
        
    def QP_callback_sub_sca(self, msg):
        self.message_sca = msg.data
        self.sca_ready = True

    def QP_callback_sub_eca(self, msg):
        self.message_eca = msg.data
        self.eca_ready = True

    def QP_callback_sub_sca_grd(self, msg):
        self.message_sca_grd = msg.data
        self.sca_grd_ready = True

    def QP_callback_sub_eca_grd(self, msg):
        self.message_eca_grd = msg.data
        self.eca_grd_ready = True


    def QP_callback_sub_state(self, msg):
        self.message_state = msg.data

    def oculus_pos_callback(self, msg):
        self.message_oculus = [msg.pose.position.x,
                             msg.pose.position.y,
                             msg.pose.position.z]

    def is_ready(self):
        return self.sca_ready and self.eca_ready and self.sca_grd_ready and self.eca_grd_ready

    def return_message(self):
        return 1, self.message_joint_position, self.message_joint_velocity, self.message_ee, self.message_state, self.message_obs1, self.message_obs2, self.message_sca, self.message_eca, self.message_sca_grd, self.message_eca_grd, self.message_ee_quat, self.message_oculus

if __name__ == "__main__":
    rospy.init_node("cvxpy_QP_all")
    rospy.logwarn("CVXPY QP Node initialized!")
    r = rospy.Rate(1500)  
    # gc.disable()
    publish_world_alias()
    Subscriber_QP = subscriber_QP()

    rospy.loginfo("Waiting for /sca_topic and /eca_topic...")
    while not rospy.is_shutdown() and not Subscriber_QP.is_ready():
        rospy.sleep(0.05)
    rospy.loginfo("Got both /sca and /eca, starting main loop!")
    
    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size=1)
    # publisher (latched so it stays visible)
    tar_pub = rospy.Publisher("/target_marker", Marker, queue_size=1, latch=True)
    # obs_pub1 = rospy.Publisher("/obstacle_marker1", Marker, queue_size=1, latch=True)
    # obs_pub2 = rospy.Publisher("/obstacle_marker2", Marker, queue_size=1, latch=True)

    lambda1, lambda2, lambda3 = 9, 10, 10
    alpha = 1e-3
    
    # q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    # q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    q_min = np.array([-2.5, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8])
    q_max = np.array([ 2.5,  1.7,  2.8, -0.1,  2.8,  3.7,  2.8])
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])*0.8
    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    
    # target_pos = np.array([0, -0.3, 0.4])
    target_pos = np.array([0.4, 0.0, 0.4])
    obs_pos = np.array([0.3, 0.0, 0.5])
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
        qe = functions_collision.compute_qe(q, q_dot)
        end_pos = Subscriber_QP.return_message()[3]
        gamma_val = Subscriber_QP.return_message()[7]
        grad_gamma = Subscriber_QP.return_message()[9]
        
        dst_sdf = Subscriber_QP.return_message()[8]
        print(f"gamma: {gamma_val:.4f}", f"dst_sdf: {dst_sdf:.4f}")
        grad_sdf = Subscriber_QP.return_message()[10]

        oculus_pos = Subscriber_QP.return_message()[12]
        
        target_pos = target_pos if oculus_pos is None else np.array(oculus_pos)
        print("target_pos:", target_pos)
        # grad_gamma = np.asarray(grad_gamma, dtype=np.float64) if grad_gamma is not None else None
        # grad_sdf   = np.asarray(grad_sdf,   dtype=np.float64) if grad_sdf   is not None else None
        grad_gamma = to_vector_or_none(grad_gamma)   # 期望长度 7 或 14；否则 None
        # print("grad_gamma:", grad_gamma)
        grad_sdf   = to_vector_or_none(grad_sdf, expect_len_set=(7,))  # 期望 7 维
        # print("grad_sdf:", grad_sdf)
        # obs_pos1 = Subscriber_QP.return_message()[5]-np.array([-2.5058, 1.0381, 0.888])
        # obs_pos2 = Subscriber_QP.return_message()[6]-np.array([-2.5058, 1.0381, 0.888])
        # print("obs_pos1:", obs_pos1)
        # print("obs_pos2:", obs_pos2)
    
        
        Jacobian_raw = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
        Jacobian = Jacobian_raw[0:3, :].T 
        Jpos = np.array(Subscriber_QP.return_message()[4][63: ]).reshape((6,7), order="F")[0:3, :]   # 3x7
        Jw   = np.array(Subscriber_QP.return_message()[4][63: ]).reshape((6,7), order="F")[3:6, :]
        # print("Jacobian:", Jacobian)
        Jpos_pinv = np.linalg.pinv(Jpos)            # 7x3
        N = np.eye(7) - Jpos_pinv @ Jpos            # 7x7
        qx, qy, qz, qw = Subscriber_QP.message_ee_quat
        R_we = tft.quaternion_matrix([qx, qy, qz, qw])[:3, :3]   # 3x3

        z_e = R_we[:, 2]                          # EE local z-axis expressed in world
        z_des = np.array([0.0, 0.0, -1.0])        # straight down
        # axis-only error: cross product gives the right minimal 3D error that ignores yaw about z_des
        e_axis = np.cross(z_e, z_des)             # 3,
        # Optional “scaled” small-angle error (robust near alignment):
        dot_ = float(np.clip(np.dot(z_e, z_des), -1.0, 1.0))
        ang  = np.arctan2(np.linalg.norm(e_axis), dot_)          # angle between z_e and z_des
        e_R  = e_axis if np.linalg.norm(e_axis) < 1e-9 else e_axis / np.linalg.norm(e_axis) * ang

        # Angular velocity at EE (to damp)
        omega = Jw @ np.asarray(q_dot)            # 3,

        # Gains (tune gently)
        K_R = 10.0         # proportional on axis-alignment (try 3–10)
        K_w = 3.0         # damping (try 1–5)

        # Nullspace orientation torque: won’t disturb 3D position task
        tau_ori = N.T @ (Jw.T @ (K_R * e_R - K_w * omega))   # 7,
        
        xdot = Jacobian.T @ np.array(q_dot)  

        MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
        Gravity = np.array(Subscriber_QP.return_message()[4][7:14])
        # target_frame = rospy.get_param("~target_frame", "panda_link0")
        # obs_pos[2] = 0.5 + 0.3 * np.sin(0.6 * time.time())

        # publish once (latched) or periodically
        tar_pub.publish(make_sphere_marker(target_pos, scale=0.04, color=(0.0, 1.0, 0.0)))
        
        soft = False
        
        

        fx = -5* (np.array(end_pos) - target_pos)
        # fx[2] =fx[2]*1.3
        # print("fx:", np.linalg.norm(fx))
        # if np.linalg.norm(fx) > 1e-2 and np.linalg.norm(fx) < 5e-1 and gamma_val > 12.6 and dst_sdf > 0.151:
        #     fx1 = fx
        #     fx = fx / (np.linalg.norm(fx)**1.2)
        #     if np.linalg.norm(fx1) < 3e-1:
        #         fx = fx / (np.linalg.norm(fx)**1)
        # print("fx after:", np.linalg.norm(fx))
        e1 = fx / (np.linalg.norm(fx)+1e-6)
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

        
        
        
        
        for idx in range(7):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4

  
        J_in = list(np.linalg.pinv(Jacobian).flatten('F'))
        fc_in = list(fc)
        M_in = list(np.array(M_inv).flatten('F'))
        b_in = list(tau_id)
        qdd_lb_in = list(qdd_lb)
        qdd_ub_in = list(qdd_ub)
        # qdd_lb_in = list(-acc_max)
        # qdd_ub_in = list(acc_max)


        if dst_sdf < 0.2:
            soft = True
            dt = 0.02
            # 1) 计算中间量
            c     = np.dot(grad_sdf, q_dot) * dt                    # grad·qd * dt
            g_eff = 0.5 * grad_sdf * dt**2                       # 0.5 * grad * dt^2
            # qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
            # tau_cmd = MassMatrix @ qdd_cmd + Coriolis
            # target_torque = list(tau_cmd)
            g_eff_in = list(g_eff)
            c_const = float(np.dot(np.asarray(grad_sdf, dtype=np.float64), np.asarray(q_dot, dtype=np.float64)) * dt)
            c_in = [c_const]
            # eps     = 0.5*(1.44 - (gamma_val*0.1)**2)
            eps     = (0.22- dst_sdf)**2
            e_in = [eps]
            try:
                target_torque = CVXsolver_sca(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in, list(g_eff), e_in, c_in)

            except Exception as e:
                rospy.logwarn("CVXsolver_eca failed")

            # print("Using soft constraint!")
            rospy.logwarn("External Collision Avoidance limits are active!")

        
        # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
        if not soft:
            if gamma_val < 14 and grad_gamma is not None and dst_sdf > 0.2:
                rospy.logwarn("Self Collision Avoidance limits are active!")
                # print("gamma:", gamma_val)
                # print("grad_gamma:", grad_gamma)
                grad_q  = grad_gamma[:7]
                grad_qd = grad_gamma[7:]
                g_eff   = 0.5 * grad_q * dt**2 + grad_qd * dt
                g_eff_in = list(g_eff)
                c_const = float(np.dot(np.asarray(grad_q, dtype=np.float64), np.asarray(q_dot, dtype=np.float64)) * dt)
                c_in = [c_const]
                eps     = ((1.405 - (gamma_val*0.1))**0.6)*0.6
                # eps     = 0.1*(1.25 - (gamma_val*0.1))
                e_in = [eps]
                try:
                    target_torque = CVXsolver_sca(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in, list(g_eff), e_in, c_in)

                except Exception as e:
                    rospy.logwarn("CVXsolver_sca failed")
                    # target_torque = list(Jacobian.T @ fc)
            else:
        # if True:
                try:
                    # start = time.time()
                    target_torque = CVXsolver_basic(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in)
                    # print(target_torque)
                    
                    # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
                    
                except Exception as e:
                    rospy.logwarn("CVXsolver_basic failed")
                    # target_torque = list(Jacobian.T @ fc)
    
        end = time.time()
            # target_torque = np.array(u.value)
            # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
            # target_torque = [-14, -14, -14, -14, -2, -2, -2]
        target_torque[0] -= 3*q[0]
        target_torque = np.asarray(target_torque, dtype=np.float64) #+ tau_ori
        target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
        msg_torque = Float32MultiArray()
        msg_torque.data = target_torque
        # print("torque:", target_torque)
        pub.publish(msg_torque)
        
        
    
    
        # r.sleep()

