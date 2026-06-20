%% Clear everything and Set Bag Directory
clear all; clc; close all

%%---> MODIFY THESE DIRECTORIES
bag_dir = '../../bags/xsens-mitsubishi-nov21/';
data_dir = '../../mat/xsens-mitsubishi-nov21/';
bags = dir(strcat(bag_dir,'*.bag'));

%% Set Topics of Interest (all of these are geometry_msgs::PoseStamped)

% From XSens IMU Motion Capture System
rh_pose_topic      = '/xsens_rh_pose';
rp_pose_topic      = '/xsens_rp_pose';
pelvis_pose_topic  = '/xsens_pelvis_pose'; % Not used, but here for completeness

% From AprilTags Tracking on Kinect Images (Static)
pick_tray_pose_topic     = '/workspace/pick_tray';
scanner_pose_topic       = '/workspace/scanner';
proc_station_pose_topic  = '/workspace/proc_station';
check_station_pose_topic = '/workspace/check_station';
ok_tray_pose_topic       = '/workspace/OK_tray';
ng_tray_pose_topic       = '/workspace/NG_tray';

%% Read Topics from N demonstrations (bags)

% Compute Relative Frames
[H_bottom_platform, H_workspace_table, H_base_link] = computeBaseLinkinXsens();

% Extract measurements from ROSBag
N = length(bags);
data_rh             = {};
data_rp             = {};
workspace_objects   = {};
for ii=1:N   
    
    % Load one bag and visualize info
    fprintf('Reading bag %s \n',bags(ii).name);
    bag = rosbag(strcat(bag_dir,bags(ii).name));
     
    % Create data structure for right hand measurements
    data_rh{ii}      = extractPoseStamped(bag, rh_pose_topic);     
    % Convert to base_link
    H_tmp = convert2H(data_rh{ii}.pose);
    for jj=1:size(H_tmp,3);H_tmp(:,:,jj) = inv(H_base_link)*H_tmp(:,:,jj);end
    data_rh{ii}.pose = convert2X(H_tmp);
    data_rh{ii}.name = 'base_link';
    
    % Create data structure for right hand prop measurements
    data_rp{ii}      = extractPoseStamped(bag, rp_pose_topic); 
    % Convert to base_link
    H_tmp = convert2H(data_rp{ii}.pose);
    for jj=1:size(H_tmp,3);H_tmp(:,:,jj) = inv(H_base_link)*H_tmp(:,:,jj);end
    data_rp{ii}.pose = convert2X(H_tmp);     
    data_rp{ii}.name = 'base_link';
    
    % Extract static workspace objects
    workspace_objects_p = zeros(7,6);
    workspace_objects_H = zeros(4,4,6);

    workspace_objects_p(:,1) = extractPoseStamped(bag, pick_tray_pose_topic).pose(1:7,1);    
    workspace_objects_p(:,3) = extractPoseStamped(bag, proc_station_pose_topic).pose(1:7,1);
    workspace_objects_p(:,4) = extractPoseStamped(bag, check_station_pose_topic).pose(1:7,1);
    workspace_objects_p(:,5) = extractPoseStamped(bag, ok_tray_pose_topic).pose(1:7,1); 
    workspace_objects_p(:,6) = extractPoseStamped(bag, ng_tray_pose_topic).pose(1:7,1); 

    % This should come from recordings but forgot to include it grrr
    workspace_objects_p(:,2) = workspace_objects_p(:,3);
    workspace_objects_p(1:3,2) = 0.5*(workspace_objects_p(1:3,3) + workspace_objects_p(1:3,4)) + [-0.30; 0.12; -0.07];
    
    % Convert to H-matrix representation
    workspace_objects_H = convert2H(workspace_objects_p);
    workspace_objects{ii}.pose = workspace_objects_p;
    workspace_objects{ii}.H = workspace_objects_H;
end

%% Visualize Trajectories on Mitsubishi Workspace!!
close all; 
figure('Color',[1 1 1])

% Plot Mitsibishi Workspace
Objects_APregions = plotMitsubishiWorkspace(H_bottom_platform, H_workspace_table, H_base_link, workspace_objects{1}.H);

% Plot Demonstrations
sample_step  = 1;
for ii=1:N
% Extract desired variables
hand_traj  = data_rp{ii}.pose(1:3,1:sample_step:end);   

% Plot Cartesian Trajectories
scatter3(hand_traj(1,:), hand_traj(2,:), hand_traj(3,:), 7.5, 'MarkerEdgeColor','k','MarkerFaceColor',[rand rand rand]); hold on;
hold on;
end

xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize',20);
ylabel('$x_2$', 'Interpreter', 'LaTex','FontSize',20);
zlabel('$x_3$', 'Interpreter', 'LaTex','FontSize',20);
title('XSens Raw Right Hand Trajectories',  'Interpreter', 'LaTex','FontSize',20)
xlim([-0.25 1.75])
ylim([-1.1 1.1])
zlim([-1  1.5])
view([62,22])
grid on
axis equal

%% Save raw data to matfile
matfile = strcat(data_dir,'demos_nov2021_raw_data.mat');
save(matfile,'data_rh','data_rp', 'workspace_objects', 'Objects_APregions', 'bags','bag_dir')
