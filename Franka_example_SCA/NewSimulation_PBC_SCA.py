import sys
import torch
import torch.nn as nn

sys.path.append('./src')

import time
import numpy as np
import cvxpy as cp
from panda_noball import Panda
from torch.autograd.functional import hessian
import matplotlib.pyplot as plt
from scipy.io import savemat
duration = 10
stepsize = 1e-3

q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

robot = Panda(stepsize)
robot.setControlMode("torque")

lambda1 = 10
lambda2 = 100
lambda3 = 100

# plt.ion()
# plt.figure(1)
# t = [0]
# t_now = 0

# class BinaryClassifier(nn.Module):
#     def __init__(self):
#         super(BinaryClassifier, self).__init__()
#         self.linear1 = nn.Linear(7, 80)
#         self.linear2 = nn.Linear(80, 50)
#         self.linear3 = nn.Linear(50, 30)
#         self.linear4 = nn.Linear(30, 10)
#         self.linear5 = nn.Linear(10, 2)
#         self.leakyrelu = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         out = self.linear1(x)
#         out = self.tanh(out)
#         out = self.linear2(out)
#         out = self.tanh(out)
#         out = self.linear3(out)
#         out = self.tanh(out)
#         out = self.linear4(out)
#         out = self.tanh(out)
#         out = self.linear5(out)
#         return out

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear1 = nn.Linear(7, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 40)
        self.linear4 = nn.Linear(40, 40)
        self.linear5 = nn.Linear(40, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        out = self.tanh(out)
        out = self.linear4(out)
        out = self.tanh(out)
        out = self.linear5(out)
        return out

device = torch.device('cuda')
# model = torch.load(f"./SCA_boundary_tanh.pt",map_location=torch.device('cpu'))
# model = torch.load(f"./SCA_boundary_tanh.pt")
model = torch.load(f"./SCA_boundary_5000.pt")

print(torch.cuda.is_available())

def f(inputs):
    return model(inputs)[1] - model(inputs)[0]

Tau_SCA = []
mindvector = []

for i in range(int(duration / stepsize)):
    if i % 1000 == 0:
        print("Simulation time: {:.3f}".format(robot.t))

    end_pos = robot.solveForwardKinematics()[0]
    # fx = [-50 * (end_pos[0] - 0.3), -50 * (end_pos[1] - 0.3), -50 * (end_pos[2] - 0.5)]
    fx = [-50 * (end_pos[0] - 0), -50 * (end_pos[1] - 0), -50 * (end_pos[2] - 0.3)]

    #### Compute damping matrix ####
    e1 = np.array(fx)/np.linalg.norm(fx)
    e2_0 = np.array([1, 0, 0])
    e3_0 = np.array([0, 1, 0])
    e2 = e2_0 - np.dot(e2_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1))
    e2 = e2/np.linalg.norm(e2)
    e3 = e3_0 - np.dot(e3_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1)) - np.dot(e3_0, (e2 / np.linalg.norm(e2))) * (e2 / np.linalg.norm(e2))
    e3 = e3/np.linalg.norm(e3)
    Q = np.zeros([3, 3])
    Q[:, 0] = np.transpose(e1)
    Q[:, 1] = np.transpose(e2)
    Q[:, 2] = np.transpose(e3)

    Lambda = np.diag([lambda1, lambda2, lambda3])

    D = Q @ Lambda @ np.transpose(Q)

    xdot = robot.getEndVelocity()

    #### Compute desired target force ####
    fc = -D @ (np.transpose(np.array(xdot)) - np.transpose(fx))

    #### Compute Jacobian ####
    Jacobian = np.zeros([7, 3])
    Jacobian[..., 0] = np.transpose(np.array(robot.getJacobian()[0]))
    Jacobian[..., 1] = np.transpose(np.array(robot.getJacobian()[1]))
    Jacobian[..., 2] = np.transpose(np.array(robot.getJacobian()[2]))
    # target_torque = np.transpose(np.linalg.pinv(Jacobian)) @ fc

    #### Compute parameters of QP ####

    ## Dynamics constraints ##
    q = robot.getJointStates()[0]
    q_dot = robot.getJointStates()[1]
    q_ddot_test = [0, 0, 0, 0, 0, 0, 0]
    tau = robot.solveInverseDynamics(q, q_dot, q_ddot_test)
    b = np.array(robot.getMassMatrix(q)) @ np.transpose(np.array(q_ddot_test)) - np.transpose(np.array(tau))

    alpha1 = 10
    alpha2 = 100

    alpha1_JL = 100
    alpha2_JL = 100

    ## Self-collision avoidance constraints ##
    start_time = time.time()
    epsilon_SCA = 10
    inputs = torch.tensor(q, dtype=torch.float32, requires_grad=True).to(device)
    output = model(inputs)[1] - model(inputs)[0]
    # dTau = torch.autograd.grad(output, inputs, retain_graph=True, create_graph=True)[0].detach().numpy()
    dTau = torch.autograd.grad(output, inputs, retain_graph=True, create_graph=True)[0].cpu().detach().numpy()
    hTau = hessian(f,inputs).cpu().detach().numpy()
    righthandside = -alpha1 * (output.cpu().detach().numpy() - epsilon_SCA) - alpha2 * dTau @ np.array(q_dot) - np.array(
        q_dot) @ hTau @ np.transpose(np.array(q_dot))
    end_time = time.time()
    Tau_SCA.append(output.cpu().detach().numpy())
    # print('SCA par cal time: ' + str(end_time - start_time))
    # print(hTau)

    epsilon_joint_limit = 0.00

    righthandside1 = -alpha1_JL*(np.array(q) - q_min - epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
    righthandside2 = -alpha1_JL*(np.array(q) - q_max + epsilon_joint_limit) - alpha2_JL*np.array(q_dot)

    #### Launch QP to get target torque ####
    x = cp.Variable(7) # Torque
    y = cp.Variable(7) # q ddot

    prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(np.transpose(np.linalg.pinv(Jacobian)))@x - fc))),
                       [np.array(robot.getMassMatrix(q))@y-x == b,
                                  dTau@y >= righthandside,
                                  y >= righthandside1,
                                  y <= righthandside2])

    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(Jacobian)@x - fc))),
    #                    [np.array(robot.getMassMatrix(q))@y-x == b,
    #                               dTau@y >= righthandside,
    #                               y >= righthandside1,
    #                               y <= righthandside2])

    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(np.transpose(np.linalg.pinv(Jacobian)))@x - fc))),
    #                 [np.array(robot.getMassMatrix(q))@y-x == b,
    #                               y >= righthandside1,
    #                               y <= righthandside2])

    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.transpose(Jacobian) @ x - fc))),
    #                                       [np.array(robot.getMassMatrix(q))@y-x == b,
    #                                        dTau@y >= righthandside])
    #prob = cp.Problem(cp.Minimize(cp.square(cp.norm2(np.linalg.pinv(Jacobian) @ x - fc))))
    # prob = cp.Problem(cp.Minimize(cp.square(cp.norm2( x - Jacobian @ fc))),
    #                    [np.array(robot.getMassMatrix(q))@y-x == b])
    prob.solve()

    target_torque = [x.value[0], x.value[1], x.value[2], x.value[3], x.value[4], x.value[5], x.value[6]]
    robot.setTargetTorques(target_torque)

    robot.step()
    mindvector.append(robot.getClosestPoints(1, 6)[0][8])

    if i % 100 == 0:
        print(robot.solveForwardKinematics())

    time.sleep(robot.stepsize)

# print(Tau_SCA)

mdic = {"a": Tau_SCA, "label": "experiment"}
savemat("Tau_SCA_active.mat", mdic)

mdic = {"a": mindvector, "label": "experiment"}
savemat("mindist_active.mat", mdic)

plt.plot(range(int(duration / stepsize)), Tau_SCA)
plt.show()
# plt.plot(range(int(duration / stepsize)), mindvector)
# plt.show()
