function [Objects_APRegions] = plotFrankaCookingWorkspace(H_left_bowl, H_center_bowl, H_right_bowl)
% Drawing fixed frames
H_origin = eye(4);
drawframe(H_origin,0.05);
hold on;

drawframe(H_left_bowl,0.05);
hold on;

drawframe(H_center_bowl,0.05);
hold on;

drawframe(H_right_bowl,0.05);
hold on;


% Draw Robot Table
P = [0.25 0 -0.025];   % you center point % Origin of TAble
L = [0.70 1.80 0.05];  % your cube dimensions 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.4, [0.9290 0.6940 0.1250]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on


% Draw Workspace Table
P = [0.55 -0.05 0.06];   % you center point % Origin of TAble
L = [0.55,0.80,0.12];  % your cube dimensions 
O = P-L/2 ;      % Get the origin of cube so that P is at center 
plotcube(L,O,.6,[0.6350 0.0780 0.1840]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on

Objects_APRegions = {};

%%%%% Draw Left Bowl
P = H_left_bowl(1:3,4)';
L = [0.21,0.21,0.06];
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

%%%%% Draw Center Bowl
P = H_center_bowl(1:3,4)';
L = [0.21,0.21,0.06];
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0 1 1]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*c')
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


%%%%% Draw Right Bowl
P = H_right_bowl(1:3,4)';
L = [0.21,0.21,0.06];
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0 0 1]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*c')
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

end
