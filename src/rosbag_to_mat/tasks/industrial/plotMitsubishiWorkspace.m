function [Objects_APRegions] = plotMitsubishiWorkspace(H_bottom_platform, H_workspace_table,  H_base_link, objects_H)
% Drawing fixed frames
H_origin = eye(4);
drawframe(H_origin,0.05);
hold on;

drawframe(H_bottom_platform,0.05);
hold on;

drawframe(H_workspace_table,0.05);
hold on;

H_xsens_origin = inv(H_base_link);
drawframe(H_xsens_origin,0.05);
hold on;

H_left_foot = eye(4);
H_left_foot(1:3,4)  = [0.0;0.05;0.0];
H_left_foot = H_xsens_origin*H_left_foot;
drawframe(H_left_foot,0.05);
hold on;

H_right_foot = eye(4);
H_right_foot(1:3,4)  = [0.0;-0.05;0.0];
H_right_foot = H_xsens_origin*H_right_foot;
drawframe(H_right_foot,0.05);
hold on;

% Draw object frames in the workspace
for i=1:size(objects_H,3)
    drawframe(objects_H(:,:,i),0.05);
    hold on;    
end

% Draw Robot Platform
P = [H_bottom_platform(1,4),H_bottom_platform(2,4), -0.835/2];   % you center point % Origin of TAble
L = [0.912 0.912 0.835];  % your cube dimensions 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0.5 0.5 0.5]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on


% Draw Workspace Table
P = H_workspace_table(1:3,4)';
L = [0.74,1.50,0.75];  % your cube dimensions 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.4,[0.5 0.5 0.5]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on


Objects_APRegions = {};
%%%%% Draw Pick Tray
L = [0.43,0.32,0.225];
P = objects_H(1:3,4,1)' - L/2  + [0 0.1 0.075]; 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0.0 0.0 1.0]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on
Objects_APRegions{1}.O = O;
Objects_APRegions{1}.L = L;
% Compute vertices for convex hull
V = zeros(3,8);
V(:,1) = O' + [0; 0; 0];
V(:,2) = O' + [L(1); 0; 0];
V(:,3) = O' + [0; L(2); 0];
V(:,4) = O' + [L(1); L(2); 0];
V(:,5) = O' + [0; 0; L(3)];
V(:,6) = O' + [L(1); 0; L(3)];
V(:,7) = O' + [0;L(2); L(3)];
V(:,8) = O' + [L(1); L(2); L(3)];
scatter3(V(1,:), V(2,:), V(3,:))
hold on;
Objects_APRegions{1}.V = V;

%%%%% Draw Scanner
L = [0.10,0.10,0.30];
P = objects_H(1:3,4,2)' - L/2 + [0.05 0.05 0.30]; 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[1.0 1.0 0.0]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on
Objects_APRegions{2}.O = O;
Objects_APRegions{2}.L = L;
% Compute vertices for convex hull
V = zeros(3,8);
V(:,1) = O' + [0; 0; 0];
V(:,2) = O' + [L(1); 0; 0];
V(:,3) = O' + [0; L(2); 0];
V(:,4) = O' + [L(1); L(2); 0];
V(:,5) = O' + [0; 0; L(3)];
V(:,6) = O' + [L(1); 0; L(3)];
V(:,7) = O' + [0;L(2); L(3)];
V(:,8) = O' + [L(1); L(2); L(3)];
scatter3(V(1,:), V(2,:), V(3,:))
hold on;
Objects_APRegions{2}.V = V;


%%%%% Draw Proc Station
L = [0.20,0.20,0.225];
P = objects_H(1:3,4,3)' + L/2  + [-0.15 0.0 -0.15]; 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[1.0 0.0 0.0]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on
Objects_APRegions{3}.O = O;
Objects_APRegions{3}.L = L;
% Compute vertices for convex hull
V = zeros(3,8);
V(:,1) = O' + [0; 0; 0];
V(:,2) = O' + [L(1); 0; 0];
V(:,3) = O' + [0; L(2); 0];
V(:,4) = O' + [L(1); L(2); 0];
V(:,5) = O' + [0; 0; L(3)];
V(:,6) = O' + [L(1); 0; L(3)];
V(:,7) = O' + [0;L(2); L(3)];
V(:,8) = O' + [L(1); L(2); L(3)];
scatter3(V(1,:), V(2,:), V(3,:))
hold on;
Objects_APRegions{3}.V = V;


%%%%% Draw Check Station
L = [0.15,0.15,0.175];
P = objects_H(1:3,4,4)' - L/2  + [0.1 -0.0 0.0575]; 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0.0 1.0 0.0]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on
Objects_APRegions{4}.O = O;
Objects_APRegions{4}.L = L;
% Compute vertices for convex hull
V = zeros(3,8);
V(:,1) = O' + [0; 0; 0];
V(:,2) = O' + [L(1); 0; 0];
V(:,3) = O' + [0; L(2); 0];
V(:,4) = O' + [L(1); L(2); 0];
V(:,5) = O' + [0; 0; L(3)];
V(:,6) = O' + [L(1); 0; L(3)];
V(:,7) = O' + [0;L(2); L(3)];
V(:,8) = O' + [L(1); L(2); L(3)];
scatter3(V(1,:), V(2,:), V(3,:))
hold on;
Objects_APRegions{4}.V = V;

%%%%% Draw OK Tray
L = [0.23,0.29,0.225];
P = objects_H(1:3,4,5)' - L/2  + [0.10 0.0 0.075]; 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0.0 0.0 1.0]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on
Objects_APRegions{5}.O = O;
Objects_APRegions{5}.L = L;
% Compute vertices for convex hull
V = zeros(3,8);
V(:,1) = O' + [0; 0; 0];
V(:,2) = O' + [L(1); 0; 0];
V(:,3) = O' + [0; L(2); 0];
V(:,4) = O' + [L(1); L(2); 0];
V(:,5) = O' + [0; 0; L(3)];
V(:,6) = O' + [L(1); 0; L(3)];
V(:,7) = O' + [0;L(2); L(3)];
V(:,8) = O' + [L(1); L(2); L(3)];
scatter3(V(1,:), V(2,:), V(3,:))
hold on;
Objects_APRegions{5}.V = V;


%%%%% Draw NG Tray
L = [0.26,0.29,0.225];
P = objects_H(1:3,4,6)' - L/2  + [0.24 0.0 0.095]; 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0.0 0.0 1.0]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on
Objects_APRegions{6}.O = O;
Objects_APRegions{6}.L = L;
% Compute vertices for convex hull
V = zeros(3,8);
V(:,1) = O' + [0; 0; 0];
V(:,2) = O' + [L(1); 0; 0];
V(:,3) = O' + [0; L(2); 0];
V(:,4) = O' + [L(1); L(2); 0];
V(:,5) = O' + [0; 0; L(3)];
V(:,6) = O' + [L(1); 0; L(3)];
V(:,7) = O' + [0;L(2); L(3)];
V(:,8) = O' + [L(1); L(2); L(3)];
scatter3(V(1,:), V(2,:), V(3,:))
hold on;
Objects_APRegions{6}.V = V;

end


% H_tray = eye(4);
% H_tray(1:3,4)  = objects_p(:,1);
% drawframe(H_tray,0.05);
% hold on;
% 
% H_proc = eye(4);
% H_proc(1:3,4)   = objects_p(:,2);
% drawframe(H_proc,0.05);
% hold on;
% 
% H_check = eye(4);
% H_check(1:3,4)  = objects_p(:,2);
% drawframe(H_check,0.05);
% hold on;
% 
% H_OK = eye(4);
% H_OK(1:3,4)  = ;
% drawframe(H_OK,0.05);
% hold on;
% 
% 
% H_table_origin = eye(4);
% H_table_origin(1:3,4)  = [0.499;0.053;0.00];
% drawframe(H_table_origin,0.05);
% hold on;
