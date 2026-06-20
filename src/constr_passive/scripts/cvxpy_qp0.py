
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
import functions_qp as functions

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


def solve_qp_with_cvxpy(J, fc, M, tau_id, qdd_lb, qdd_ub, alpha=1e-2):
    u = cp.Variable(7)
    M_inv = np.linalg.inv(M)
    
    J_in = np.linalg.pinv(J)
    objective = cp.sum_squares(J_in @ u - fc) + alpha * cp.sum_squares(u)
    
    constraints = [
        M_inv @ u >= qdd_lb + M_inv @ tau_id,
        M_inv @ u <= qdd_ub + M_inv @ tau_id,
    ]
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status == cp.OPTIMAL:
            return np.array(u.value), True
        else:
            rospy.logwarn(f"QP not optimal, status: {prob.status}")
            return None, False
    except cp.SolverError as e:
        rospy.logwarn(f"QP solver error: {e}")
        return None, False


if __name__ == "__main__":
    rospy.init_node("cvxpy_QP")
    rospy.logwarn("CVXPY QP Node initialized!")
    r = rospy.Rate(500)  
    
    Subscriber_QP = subscriber_QP()
    
    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size=10)
    
    # Publishers for joint limit monitoring
    pub_joint_pos = rospy.Publisher("/joint_limit_monitor/joint_positions", Float32MultiArray, queue_size=10)
    pub_joint_min = rospy.Publisher("/joint_limit_monitor/joint_min", Float32MultiArray, queue_size=10)
    pub_joint_max = rospy.Publisher("/joint_limit_monitor/joint_max", Float32MultiArray, queue_size=10)
    pub_joint_margins = rospy.Publisher("/joint_limit_monitor/joint_margins", Float32MultiArray, queue_size=10)
    pub_qdd_bounds = rospy.Publisher("/joint_limit_monitor/qdd_bounds", Float32MultiArray, queue_size=10)
    pub_qreal_min = rospy.Publisher("/joint_limit_monitor/qreal_min", Float32MultiArray, queue_size=10)
    pub_qreal_max = rospy.Publisher("/joint_limit_monitor/qreal_max", Float32MultiArray, queue_size=10)
    lambda1, lambda2, lambda3 = 8, 48, 48
    alpha = 1e-4
    
    q_real_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_real_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 1.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    
    target_pos = np.array([0.5, 0.0, 0.4])
    dt = 0.02  

    rospy.sleep(1.0)

    while not rospy.is_shutdown():

        q = Subscriber_QP.return_message()[1]
        q_dot = Subscriber_QP.return_message()[2]
        end_pos = Subscriber_QP.return_message()[3]
    
        
        Jacobian_raw = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
        Jacobian = Jacobian_raw[0:3, :].T 
        
        xdot = Jacobian.T @ np.array(q_dot)  

        MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
        Gravity = np.array(Subscriber_QP.return_message()[4][7:14])

        fx = -8 * (np.array(end_pos) - target_pos)
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
        
        
        
        M_inv = np.linalg.inv(MassMatrix)           
        qdd_zero = np.zeros(7)
        tau_id = MassMatrix @ qdd_zero + Coriolis #+ Gravity 
        
        for idx in range(7):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4
        
        target_torque, success = solve_qp_with_cvxpy(
            Jacobian, fc, MassMatrix, tau_id, qdd_lb, qdd_ub, alpha
        )
        
        
        target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
        
        
        
        msg_torque = Float32MultiArray()
        msg_torque.data = target_torque.tolist()
        pub.publish(msg_torque)
        
        # Publish joint limit monitoring data
        msg_pos = Float32MultiArray()
        msg_pos.data = q
        pub_joint_pos.publish(msg_pos)
        
        msg_min = Float32MultiArray()
        msg_min.data = q_min.tolist()
        pub_joint_min.publish(msg_min)
        
        msg_max = Float32MultiArray()
        msg_max.data = q_max.tolist()
        pub_joint_max.publish(msg_max)
        
        msg_margins = Float32MultiArray()
        joint_margins = np.minimum(q - q_min, q_max - q)
        msg_margins.data = joint_margins.tolist()
        pub_joint_margins.publish(msg_margins)
        
        msg_qdd = Float32MultiArray()
        msg_qdd.data = qdd_lb.tolist() + qdd_ub.tolist()  
        pub_qdd_bounds.publish(msg_qdd)

        msg_qreal_min = Float32MultiArray()
        msg_qreal_min.data = q_real_min.tolist()
        pub_qreal_min.publish(msg_qreal_min)

        msg_qreal_max = Float32MultiArray()
        msg_qreal_max.data = q_real_max.tolist()
        pub_qreal_max.publish(msg_qreal_max)
            
            

        
        r.sleep()
