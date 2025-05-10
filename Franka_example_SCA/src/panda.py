import pybullet as p

class Panda:
    def __init__(self, stepsize=1e-3, realtime=0):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque"

        #self.position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        #self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self.position_control_gain_p = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self.max_torque = [10000,10000,10000,10000,10000,10000,10000]

        # connect pybullet
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        #p.setGravity(0,0,-10)
        p.setGravity(0, 0, 0)

        # load models
        p.setAdditionalSearchPath("../models")

        # self.plane = p.loadURDF("./plane/plane.urdf",
        #                         useFixedBase=True)
        self.plane = p.loadURDF("/home/zhiquan/Code/simulation7.4/franka-pybullet-master/franka-pybullet-master/models/plane/plane.urdf",
                                useFixedBase=True)
        p.changeDynamics(self.plane,-1,restitution=.95)
        # self.robot = p.loadURDF("panda/panda.urdf",
        #                         useFixedBase=True,
        #                         flags=p.URDF_USE_SELF_COLLISION)
        self.robot = p.loadURDF("/home/zhiquan/Code/simulation7.4/franka-pybullet-master/franka-pybullet-master/models/panda/panda.urdf",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)
        # ball
        # self.ball = p.loadURDF("ball/ball.urdf")
        self.ball = p.loadURDF("/home/zhiquan/Code/simulation7.4/franka-pybullet-master/franka-pybullet-master/models/ball/ball.urdf")
        p.changeDynamics(self.ball, -1, restitution=.95, linearDamping=1e-2, angularDamping=1e-2)

        self.default_ball_pos = [0.35, -0.15, 0.7]
        self.default_ball_ori = [0, 0, 0, 1]
        p.resetBasePositionAndOrientation(self.ball, self.default_ball_pos, self.default_ball_ori)

        # # robot parameters
        self.dof = p.getNumJoints(self.robot) - 1 # Virtual fixed joint between the flange and last link
        # if self.dof != 7:
        #     raise Exception('wrong urdf file: number of joints is not 7')

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
            self.target_pos.append((self.q_min[j] + self.q_max[j])/2.0)
            self.target_torque.append(0.)

        self.reset()

    def reset(self):
        self.t = 0.0        
        self.control_mode = "torque"
        for j in range(self.dof):
            self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            self.target_torque[j] = 0.
            p.resetJointState(self.robot,j,targetValue=self.target_pos[j])

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
                                    #forces=self.max_torque,
                                    #positionGains=self.position_control_gain_p,
                                    #velocityGains=self.position_control_gain_d
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
        #velocity = p.getLinkState(self.robot, 7, True)[6]
        #ori = p.getLinkState(self.robot, 7)[1]
        #print(p.getJointState(self.robot, 0))
        jointpos = [0, 0, 0, 0, 0, 0, 0]
        jointpos[0] = p.getJointState(self.robot, 0)[0]
        jointpos[1] = p.getJointState(self.robot, 1)[0]
        jointpos[2] = p.getJointState(self.robot, 2)[0]
        jointpos[3] = p.getJointState(self.robot, 3)[0]
        jointpos[4] = p.getJointState(self.robot, 4)[0]
        jointpos[5] = p.getJointState(self.robot, 5)[0]
        jointpos[6] = p.getJointState(self.robot, 6)[0]
        jacobian = p.calculateJacobian(self.robot, 7, [0, 0, 0],jointpos ,[0, 0, 0, 0, 0, 0, 0] , [0, 0, 0, 0, 0, 0, 0])[0]
        return jacobian

    def getJacobian_ori(self):
        #velocity = p.getLinkState(self.robot, 7, True)[6]
        #ori = p.getLinkState(self.robot, 7)[1]
        #print(p.getJointState(self.robot, 0))
        jointpos = [0, 0, 0, 0, 0, 0, 0]
        jointpos[0] = p.getJointState(self.robot, 0)[0]
        jointpos[1] = p.getJointState(self.robot, 1)[0]
        jointpos[2] = p.getJointState(self.robot, 2)[0]
        jointpos[3] = p.getJointState(self.robot, 3)[0]
        jointpos[4] = p.getJointState(self.robot, 4)[0]
        jointpos[5] = p.getJointState(self.robot, 5)[0]
        jointpos[6] = p.getJointState(self.robot, 6)[0]
        jacobian = p.calculateJacobian(self.robot, 7, [0, 0, 0],jointpos ,[0, 0, 0, 0, 0, 0, 0] , [0, 0, 0, 0, 0, 0, 0])[1]
        return jacobian

    def getJacobianarbitary(self, jointstate):
        #velocity = p.getLinkState(self.robot, 7, True)[6]
        #ori = p.getLinkState(self.robot, 7)[1]
        #print(p.getJointState(self.robot, 0))
        jointpos = [0, 0, 0, 0, 0, 0, 0]
        jointpos[0] = p.getJointState(self.robot, 0)[0]
        jointpos[1] = p.getJointState(self.robot, 1)[0]
        jointpos[2] = p.getJointState(self.robot, 2)[0]
        jointpos[3] = p.getJointState(self.robot, 3)[0]
        jointpos[4] = p.getJointState(self.robot, 4)[0]
        jointpos[5] = p.getJointState(self.robot, 5)[0]
        jointpos[6] = p.getJointState(self.robot, 6)[0]
        jacobian = p.calculateJacobian(self.robot, 7, [0, 0, 0],jointstate ,[0, 0, 0, 0, 0, 0, 0] , [0, 0, 0, 0, 0, 0, 0])[0]
        return jacobian

    def getMassMatrix(self, jointstates):
        mass = p.calculateMassMatrix(self.robot, jointstates)
        return mass

    def getBallStates(self):
        ball_states = p.getBasePositionAndOrientation(self.ball)
        ball_pos = list(ball_states[0])
        ball_ori = list(ball_states[1])
        return ball_pos, ball_ori

if __name__ == "__main__":
    robot = Panda(realtime=1)
    while True:
        pass
