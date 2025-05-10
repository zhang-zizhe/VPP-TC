clc
clear
close all

addpath("functions/")
load('meshes/mesh_full.mat');
load('data_SCA_new.mat');
q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0];

base = eye(4);
r = [0 0 0 0.0825 -0.0825 0 0.088 0];
d = [0.333 0 0.316 0 0.384 0 0 0.107];
alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];

joint_state = [-0.0396455856573685,1.63806336837627,0.00376656300542422,-2.76631310763505,1.33246714240137,2.28583708948343,2.08768133973368,0];
P = franka_dh_fk(joint_state, r, d, alpha,base);

i = 1;
j = 2;
k = 3;
t = 4;
t1 = 5;
t2 = 6;
t3 = 7;
t4 = 8;

R1 = P{i}(1:3,1:3);
T1 = P{i}(1:3,4);
V1 = mesh{i}.v*R1'+T1';
dotid1 = mesh{i}.f;

R2 = P{j}(1:3,1:3);
T2 = P{j}(1:3,4);
V2 = mesh{j}.v*R2'+T2';
dotid2 = mesh{j}.f;

R3 = P{k}(1:3,1:3);
T3 = P{k}(1:3,4);
V3 = mesh{k}.v*R3'+T3';
dotid3 = mesh{k}.f;

R4 = P{t}(1:3,1:3);
T4 = P{t}(1:3,4);
V4 = mesh{t}.v*R4'+T4';
dotid4 = mesh{t}.f;

R5 = P{t1}(1:3,1:3);
T5 = P{t1}(1:3,4);
V5 = mesh{t1}.v*R5'+T5';
dotid5 = mesh{t1}.f;

R6 = P{t2}(1:3,1:3);
T6 = P{t2}(1:3,4);
V6 = mesh{t2}.v*R6'+T6';
dotid6 = mesh{t2}.f;

R7 = P{t3}(1:3,1:3);
T7 = P{t3}(1:3,4);
V7 = mesh{t3}.v*R7'+T7';
dotid7 = mesh{t3}.f;

R8 = P{t4}(1:3,1:3);
T8 = P{t4}(1:3,4);
V8 = mesh{t4}.v*R8'+T8';
dotid8 = mesh{t4}.f;
% COMPUTE MINIMUM DISTANCE AND RETURN VALUE
% tic
% dist = openGJK( V2', V7' ); 
% toc
% fprintf('The minimum distance between A and B is %.8f\n',dist);


% VISUALISE RESULTS ONLY IN MATLAB
if(exist('OCTAVE_VERSION', 'builtin') == 0)
       % .. create new figure
       figure('units','centimeters', 'WindowStyle','normal', 'color','w',...
       'Position',[0 8.5 9 6],'defaultAxesColorOrder',parula,...
       'Renderer','opengl') 
       % .. adjust properties
       axis equal tight off; hold all; 
       % .. display body A
       DT = delaunayTriangulation(V1);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .9 ],...
              'FaceLighting','flat' )
       % .. display body B
       DT = delaunayTriangulation(V2);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .8 ],...
              'FaceLighting','flat' )
         
       DT = delaunayTriangulation(V3);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .7 ],...
              'FaceLighting','flat' )
          
       DT = delaunayTriangulation(V4);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .6 ],...
              'FaceLighting','flat' )
          
       DT = delaunayTriangulation(V5);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .5 ],...
              'FaceLighting','flat' )
          
       DT = delaunayTriangulation(V6);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .4 ],...
              'FaceLighting','flat' )

       DT = delaunayTriangulation(V7);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .3 ],...
              'FaceLighting','flat' )

       DT = delaunayTriangulation(V8);
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .2 ],...
              'FaceLighting','flat' )
       % .. represent the computed distance as a sphere
%        [x,y,z] = sphere(100);
%        surf(x.*dist/2,y.*dist/2,z.*dist/2,'facecolor',[.9 .9 .9],...
%        'EdgeColor','none','FaceLighting','flat','SpecularColorReflectance',0,...
%        'SpecularStrength',1,'SpecularExponent',10,'facealpha',.7)
       % ... adjust point of view   
       view(42,21)
       % ... add light
       light('Position',[5 -10 20],'Style','local'); 
end



