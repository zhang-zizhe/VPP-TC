function [Objects_APRegions] = plotFrankaInspectionWorkspace(H_pickup_station, H_inspection_tunnel, H_release_station, is_museum)
% Drawing fixed frames
H_origin = eye(4);
drawframe(H_origin,0.05);
hold on;

drawframe(H_pickup_station,0.05);
hold on;

drawframe(H_inspection_tunnel,0.05);
hold on;

drawframe(H_release_station,0.05);
hold on;


%%%%%  Draw Robot Table
P = [0.35 0 -0.025];   % you center point % Origin of TAble
L = [0.76 1.50 0.05];  % your cube dimensions 
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.4, [0.9290 0.6940 0.1250]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*k')
hold on


%%%%%  Draw Inspection Tunnel
if is_museum
    P = H_inspection_tunnel(1:3,4)' + [0 0 0.07];
    L = [0.12, 0.62, 0.14];
else
    P = H_inspection_tunnel(1:3,4)' + [0 0 0.025];
    L = [0.075,0.475,0.06 + 0.05];
end
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0 1 1]);   % use function plotcube 
hold on
plot3(P(1),P(2),P(3),'*c')
hold on


Objects_APRegions = {};

%%%%% Draw Pickup Station
if is_museum
    P = H_pickup_station(1:3,4)' + [0 0 0.13];
    L = [0.25,0.25, 0.26];
else
    P = H_pickup_station(1:3,4)' + [0 0 0.15];
    L = [0.22,0.22, 0.06 + 0.3];
end

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


%%%%% Draw Release Station
if is_museum
    P = H_release_station(1:3,4)' + [0 0 0.17];
    L = [0.14,0.14,0.32];
else
    P = H_release_station(1:3,4)' + [0 0 0.15];
    L = [0.22,0.22,0.06 + 0.3];
end
O = P-L/2 ;       % Get the origin of cube so that P is at center 
plotcube(L,O,.2,[0 0 1]);   % use function plotcube 
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

end
