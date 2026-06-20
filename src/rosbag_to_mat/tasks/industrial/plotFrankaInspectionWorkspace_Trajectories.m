function [Objects_APregions, fhandle] = plotFrankaInspectionWorkspace_Trajectories(ee_traj, is_museum, show_robot, varargin)
    if isempty(varargin)
        fhandle = figure('Color',[1 1 1], 'Position',[73   551   684   411]);
    else
        pos = varargin{1};
        fhandle = figure('Color',[1 1 1], 'Position',pos);
    end

    % Visualize Robot Arm
    if show_robot
        robot = importrobot('frankaEmikaPanda.urdf');
        show(robot,'visuals','on','collision','on');
        hold on;
    end

    % Get Station Transforms (These should be pre-defined and calibrated before running this script)
    [H_pickup_station, H_inspection_tunnel, H_release_station] = computeFrankaInspectionTransforms(is_museum);
    
    % Plot Franka Inspection Workspace
    Objects_APregions = plotFrankaInspectionWorkspace(H_pickup_station, H_inspection_tunnel, H_release_station, is_museum);  
    
    % Plot Cartesian Trajectories
    if ~isempty(ee_traj)
        scatter3(ee_traj(1,:), ee_traj(2,:), ee_traj(3,:), 7.5, 'MarkerEdgeColor','k','MarkerFaceColor',[rand rand rand]); hold on;
        hold on;
        title('Franka End-Effector Trajectories',  'Interpreter', 'LaTex','FontSize',20)
    end
    xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize',20);
    ylabel('$x_2$', 'Interpreter', 'LaTex','FontSize',20);
    zlabel('$x_3$', 'Interpreter', 'LaTex','FontSize',20);
    xlim([-0.25 1.75])
    ylim([-1.1 1.1])
    zlim([-1  1.5])
    view([113,17])
    grid on
    axis equal


end