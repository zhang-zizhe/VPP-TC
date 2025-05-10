import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear1 = nn.Linear(7, 80)
        self.linear2 = nn.Linear(80, 50)
        self.linear3 = nn.Linear(50, 30)
        self.linear4 = nn.Linear(30, 10)
        self.linear5 = nn.Linear(10, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.linear1(x)
        out = self.leakyrelu(out)
        out = self.linear2(out)
        out = self.leakyrelu(out)
        out = self.linear3(out)
        out = self.leakyrelu(out)
        out = self.linear4(out)
        out = self.leakyrelu(out)
        out = self.linear5(out)
        return out

device = torch.device('cpu')
for loop in range(2000):
    # model = torch.load("./NN_result/SCA_boundary_100.pt").to(device)
    model = torch.load(f"./NN_result/SCA_boundary_{loop+1}.pt").to(device)
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([2.8973 ,  1.7628,	 2.8973, -0.0698,  2.8973,  3.7525,	 2.8973])

    q_mid = (q_min + q_max)/2
    q1_idx = 0
    q2_idx = 1
    resolution = 100

    q0 = q_min[0]
    q1 = q_min[1]
    q2 = q_min[2]
    q3 = q_min[3]
    q4 = q_min[4]
    q5 = q_min[5]
    q6 = q_min[6]

    x = np.linspace(q_min[q1_idx], q_max[q1_idx], resolution)
    y = np.linspace(q_min[q2_idx], q_max[q2_idx], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    for i in range(len(x)):
        for j in range(len(y)):
            # inputs = torch.tensor([q0, x[i], y[j], q3, q4, q5, q6], dtype=torch.float32).to(device)
            inputs = torch.tensor([x[i], q_min[1], y[j], q_min[3], q_min[4], q_min[5], q_min[6]], dtype=torch.float32).to(device)
            # inputs = torch.tensor([q_min[0], x[i], q_min[2], y[j], q_min[4], q_min[5], q_min[6]], dtype=torch.float32).to(device)
            Z[i, j] = model(inputs)[1].item() - model(inputs)[0].item()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    plt.contourf(X, Y, Z, alpha=1)
    plt.colorbar()
    plt.xlabel('Link0')
    plt.ylabel('Link2')
    plt.title('Link0-Link2')
    plt.savefig(f'./SCA_boundary_vis/link0-link2/1_01_{loop+1}.png', dpi=600)
    # plt.show()
    plt.close()
    print(loop)
# start_time = time.time()
# inputs = torch.tensor([q_min[0], q_min[1], q_min[2], q_min[3], q_min[4], q_min[5], q_min[6]], dtype=torch.float32, requires_grad=True).to(device)
# output = model(inputs)[1]
#
# model.zero_grad()
# output.backward()
#
# # gradients = inputs.grad
# # end_time = time.time()
# # print(gradients)
# # print(end_time - start_time)
# newgradient = np.zeros((1, 7))[0]
# for i in range(7):
#     addvector = torch.zeros((1, 7))[0]
#     addvector[i] = 0.0001
#     newinput = addvector + inputs
#     newoutput = model(newinput)[1]
#     newgradient[i] = (newoutput - output)/0.0001
# end_time = time.time()
# print(newgradient)
# print(end_time - start_time)