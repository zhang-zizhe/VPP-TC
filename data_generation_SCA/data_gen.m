clc
clear
close all

addpath("functions/")
load('meshes/mesh_full.mat');

q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0];

base = eye(4);
r = [0 0 0 0.0825 -0.0825 0 0.088 0];
d = [0.333 0 0.316 0 0.384 0 0 0.107];
alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];

tic
q1 = q_min(1):0.5:q_max(1);
q2 = q_min(2):0.5:q_max(2);
q3 = q_min(3):0.5:q_max(3);
q4 = q_min(4):0.5:q_max(4);
q5 = q_min(5):0.5:q_max(5);
q6 = q_min(6):0.5:q_max(6);
q7 = q_min(7):0.5:q_max(7);

q1 = q1 + (q_max(1) - q1(length(q1)))/2;
q2 = q2 + (q_max(2) - q2(length(q2)))/2;
q3 = q3 + (q_max(3) - q3(length(q3)))/2;
q4 = q4 + (q_max(4) - q4(length(q4)))/2;
q5 = q5 + (q_max(5) - q5(length(q5)))/2;
q6 = q6 + (q_max(6) - q6(length(q6)))/2;
q7 = q7 + (q_max(7) - q7(length(q7)))/2;

joint_states = zeros(length(q1)*length(q2)*length(q3)*length(q4)*length(q5)*length(q6)*length(q7), 8);
for i1 = 1:length(q1)
    for i2 = 1:length(q2)
        for i3 = 1:length(q3)
            for i4 = 1:length(q4)
                for i5 = 1:length(q5)
                    for i6 = 1:length(q6)
                        for i7 = 1:length(q7)
                            joint_states((i1-1)*length(q7)*length(q6)*length(q5)*length(q4)*length(q3)*length(q2) + (i2-1)*length(q7)*length(q6)*length(q5)*length(q4)*length(q3) + (i3-1)*length(q7)*length(q6)*length(q5)*length(q4) + (i4-1)*length(q7)*length(q6)*length(q5) + (i5-1)*length(q7)*length(q6) + (i6-1)*length(q7) + i7, :) = [q1(i1) q2(i2) q3(i3) q4(i4) q5(i5) q6(i6) q7(i7) 0];
                        end
                    end
                end
            end
        end
    end
end
toc

niter = length(joint_states);
niter = 1;
data_SCA = zeros(niter, 29);
tic
for iter = 1:niter
    base = eye(4);
    r = [0 0 0 0.0825 -0.0825 0 0.088 0];
    d = [0.333 0 0.316 0 0.384 0 0 0.107];
    alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];
    joint_state = joint_states(iter, :);

    P = franka_dh_fk(joint_state, r, d, alpha,base);
    mindvector = [];

    for i = 1:6
        for j = (i + 2) : 8
            R1 = P{i}(1:3,1:3);
            T1 = P{i}(1:3,4);
            V1 = mesh{i}.v*R1'+T1';

            R2 = P{j}(1:3,1:3);
            T2 = P{j}(1:3,4);
            V2 = mesh{j}.v*R2'+T2';
            
            mindvector = [mindvector openGJK( V1', V2' )];
        end
    end
    data_SCA(iter, :) = [joint_state mindvector];
    if mod(iter, 1000) == 0
        disp([num2str(iter) ,'-th data generated!']);
    end
end
toc




