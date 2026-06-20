function [H_bottom_platform, H_workspace_table, H_base_link] = computeBaseLinkinXsens()
% Base link in Xsens Origin
H03=eye(4); H03(1:3,4) = [0.0 0.0 0.01]; % base_link
H02=eye(4); H02(1:3,4) = [0.251 -0.046 0.0];
H01=eye(4); H01(1:3,4) = [0.0 0.0 0.835];
H0=eye(4); H0(1:3,4) = [0.861 -0.045 0.0]; % workspace_table link
H1= eye(4); H1(1:3,4) = [0.475, 0.0, 0.0];


H_top_platform = H1*H0*H01; 
H_top_platform(1:3,1:3) = rotationMatrx('z',deg2rad(180)); % MATLAB/XSENS ORIGIN

H_base_link = H_top_platform*H02*H03;
H_bottom_platform = inv(H_base_link)*H1*H0;

% workspace_table link
H1= eye(4); H1(1:3,4) = [0.475, 0.0, 0.375];
H_workspace_table = inv(H_base_link)*H1;

end

