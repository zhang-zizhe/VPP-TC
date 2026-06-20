%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Script for Extracting Topics from Latest Recorded Bag         %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all

%%%% Set directories (if recordings are not at the same level as catkin_ws
%%%% then this should change!
bag_dir = '../../../museum_recordings/bags/';
data_dir = '../../../museum_recordings/mat/';
bags = dir(strcat(bag_dir,'*.bag'));
latest_bag  = bags(end);
do_plot     = 1;
sample_step = 20; % This downsampling is only for visualization!
show_robot   = 1;
is_museum    = 1; %1: MIT Museum Setup, 0: PENN Figueroa Lab Setup

%%%% Set Topics of Interest (all of these are geometry_msgs::PoseStamped)
ee_pose_topic         = '/franka_state_controller/O_T_EE';
gripper_joints_topic  = '/franka_gripper/joint_states'; 
% Can be used to aid segmentation, not using it now but should!

%%%% Read Topics Latest Bag
% Load one bag and visualize info
[~, bagname, ~] = fileparts(latest_bag.name);
fprintf('Reading bag %s \n',bagname);
bag = rosbag(strcat(bag_dir,latest_bag.name));
% Create data structure for right hand measurements
data_ee_pose = extractPoseStamped(bag, ee_pose_topic); 

%% Visualize Trajectories on Franka Inspection Workspace!!
if do_plot 
    % Sub-sampled raw trajectories for visualization
    ee_traj  = data_ee_pose.pose(1:3,1:sample_step:end); 
    plotFrankaInspectionWorkspace_Trajectories(ee_traj, is_museum, show_robot)
end

%% Save raw data to matfile
matfilename = strcat(bagname,'.mat');
matfile = strcat(data_dir,matfilename);
save(matfile,'data_ee_pose','bags','bag_dir')
