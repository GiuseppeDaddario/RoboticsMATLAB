function plot_robot(A, joint_vars, joint_vals)
    % Draws the robot based on symbolic A matrices (from DK),
    % substituting joint values before plotting.
    %
    % A         : cell array of symbolic 4x4 homogeneous matrices from DK
    % joint_vars: symbolic variables used in A (e.g., [theta1 theta2 ...])
    % joint_vals: numeric values for joint variables (same size as joint_vars)

    N = length(A);
    T = eye(4);  % Base frame
    positions = zeros(3, N+1);  % Store positions (including base)

    figure; hold on; grid on; axis equal;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Robot Configuration');
    view(3);

    % Plot base coordinate frame
    plot_coord_frame(T, 0.1);

    % Plot each joint
    for i = 1:N
        Ti_sym = A{i};
        Ti_num = double(subs(Ti_sym, joint_vars, joint_vals));  % Evaluate symbolically
        T = T * Ti_num;  % Update global transform
        positions(:,i+1) = T(1:3,4);  % Store joint position
        plot_coord_frame(T, 0.1);
    end

    % Plot links
    plot3(positions(1,:), positions(2,:), positions(3,:), '-o', ...
          'LineWidth', 2, 'MarkerSize', 6, 'Color', 'b');

    legend('Robot Links');
end

function plot_coord_frame(T, scale)
    % Plot a 3D coordinate frame given a homogeneous transform T
    origin = T(1:3,4);
    x_axis = T(1:3,1) * scale;
    y_axis = T(1:3,2) * scale;
    z_axis = T(1:3,3) * scale;

    quiver3(origin(1), origin(2), origin(3), x_axis(1), x_axis(2), x_axis(3), ...
            'r', 'LineWidth', 1.5);
    quiver3(origin(1), origin(2), origin(3), y_axis(1), y_axis(2), y_axis(3), ...
            'g', 'LineWidth', 1.5);
    quiver3(origin(1), origin(2), origin(3), z_axis(1), z_axis(2), z_axis(3), ...
            'b', 'LineWidth', 1.5);
end