clc
clear
close all
% 
% addpath("functions/")
% load('meshes/mesh_full.mat');
% 
% q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
% q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0];
% 
% base = eye(4);
% r = [0 0 0 0.0825 -0.0825 0 0.088 0];
% d = [0.333 0 0.316 0 0.384 0 0 0.107];
% alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];
% 
% ndata = 300000;
% collision_per = 0.5;          % class 1
% closetocollision_per = 0.35;  % class 2
% uncollided_per = 0.15;        % class 3
% 
% class1_ended_flag = 0;
% class2_ended_flag = 0;
% class3_ended_flag = 0;
% 
% nclass1 = 0;
% nclass2 = 0;
% nclass3 = 0;
% 
% data_SCA = zeros(ndata, 30);
% idx = 1;
% nloop = 0;
% 
% 
% while ((class1_ended_flag == 0)||(class2_ended_flag == 0)||(class3_ended_flag == 0))
%     joint_state = q_min + rand(1, 8).*(q_max - q_min);
%     P = franka_dh_fk(joint_state, r, d, alpha,base);
%     mindvector = [];
% 
%     for i = 1:6
%         for j = (i + 2) : 8
%             R1 = P{i}(1:3,1:3);
%             T1 = P{i}(1:3,4);
%             V1 = mesh{i}.v*R1'+T1';
% 
%             R2 = P{j}(1:3,1:3);
%             T2 = P{j}(1:3,4);
%             V2 = mesh{j}.v*R2'+T2';
%             
%             mindvector = [mindvector openGJK( V1', V2' )];
%         end
%     end
%     mind = mindvector;
%     mind(21) = [];
%     if (min(mind) == 0) && (class1_ended_flag == 0)
%         data_SCA(idx, :) = [joint_state mindvector 1];
%         idx = idx + 1;
%         nclass1 = nclass1 + 1;
%     elseif ((min(mind)>0) && (min(mind)<=0.04)) && (class2_ended_flag == 0)
%         data_SCA(idx, :) = [joint_state mindvector 2];
%         idx = idx + 1;
%         nclass2 = nclass2 + 1;
%     elseif (class3_ended_flag == 0)
%         data_SCA(idx, :) = [joint_state mindvector 3];
%         idx = idx + 1;
%         nclass3 = nclass3 + 1;
%     end
%     
%     if (nclass1 == (ndata*collision_per))
%         class1_ended_flag = 1;
%     end
%     
%     if (nclass2 == (ndata*closetocollision_per))
%         class2_ended_flag = 1;
%     end
%     
%     if (nclass3 == (ndata*uncollided_per))
%         class3_ended_flag = 1;
%     end
%     
%     nloop = nloop + 1;
%     
%     if (mod(nloop, 1000) == 0)
%         disp(['----- Class 1 ', num2str(nclass1/(ndata*collision_per)*100, 6), '% completed -----', ' Class 2 ', num2str(nclass2/(ndata*closetocollision_per)*100, 6), '% completed -----', ' Class 3 ', num2str(nclass3/(ndata*uncollided_per)*100, 6), '% completed -----']);
%     end
% end

% clc
% clear
% close all
% 
% load('data_SCA2.mat');
% ndata = 700000;
% collision_per = 0.5;          % class 1
% closetocollision_per = 0.35;  % class 2
% uncollided_per = 0.15;        % class 3
% data_SCA1 = zeros(ndata, 30);
% 
% class1_ended_flag = 0;
% class2_ended_flag = 0;
% class3_ended_flag = 0;
% 
% nclass1 = 0;
% nclass2 = 0;
% nclass3 = 0;
% 
% idx = 1;
% 
% for i = 1:length(data_SCA)
%     if (data_SCA(i, 30) == 1) && (class1_ended_flag == 0)
%         data_SCA1(idx, :) = data_SCA(i, :);
%         idx = idx + 1;
%         nclass1 = nclass1 + 1;
%     elseif (data_SCA(i, 30) == 2) && (class2_ended_flag == 0)
%         data_SCA1(idx, :) = data_SCA(i, :);
%         idx = idx + 1;
%         nclass2 = nclass2 + 1;
%     elseif (class3_ended_flag == 0)
%         data_SCA1(idx, :) = data_SCA(i, :);
%         idx = idx + 1;
%         nclass3 = nclass3 + 1;
%     end
%     
%     if (nclass1 >= (ndata*collision_per))
%         class1_ended_flag = 1;
%     end
%     
%     if (nclass2 >= (ndata*closetocollision_per))
%         class2_ended_flag = 1;
%     end
%     
%     if (nclass3 >= (ndata*uncollided_per))
%         class3_ended_flag = 1;
%     end
% end
% 

% load('data_SCA1.mat');
% data_SCA_new = [data_SCA; data_SCA1];
% data_SCA_classification = zeros(length(data_SCA_new), 8);
% for i = 1:length(data_SCA_new)
%     if data_SCA_new(i, 30) == 1
%         data_SCA_classification(i, :) = [data_SCA_new(i, 1:7), 0];
%     else
%         data_SCA_classification(i, :) = [data_SCA_new(i, 1:7), 1];
%     end
% end

% load('data_SCA_more.mat');
% data_SCA_more_new = data_SCA_more;
% data_SCA_more_classification = zeros(length(data_SCA_more_new), 8);
% for i = 1:length(data_SCA_more_new)
%     if data_SCA_more_new(i, 30) == 1
%         data_SCA_more_classification(i, :) = [data_SCA_more_new(i, 1:7), 0];
%     else
%         data_SCA_more_classification(i, :) = [data_SCA_more_new(i, 1:7), 1];
%     end
% end

load('data_SCA_more_classification.mat');
data_SCA_more_biclass = [data_SCA_more_classification(:, 1:7) zeros(length(data_SCA_more_classification), 2)];
for i = 1:length(data_SCA_more_classification)
    if (data_SCA_more_classification(i, 8) == 0)
        data_SCA_more_biclass(i, 8) = 1;
        data_SCA_more_biclass(i, 9) = 0;
    else
        data_SCA_more_biclass(i, 8) = 0;
        data_SCA_more_biclass(i, 9) = 1;
    end
end






