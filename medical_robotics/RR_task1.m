clear; clc; close all;

%% Initial configuration
% Here there are the definitions of the robot parameters (q1,q2,lambda) in the initial
% configuration and the desired orientation. Since the robot has unitary
% link lenght, l1 and l2 are not multiplied in the formulas

q0 = [pi/4; 0];
lambda0 = 0.5;
theta_d = pi/8;

p1 = [cos(q0(1)); sin(q0(1))];
p2 = [cos(q0(1)) + cos(q0(1) + q0(2)); sin(q0(1)) + sin(q0(1) + q0(2))];
p_trocar = p1 + lambda0 * (p2 - p1);

%% Integration
% Here the chosen gains are set; The integration, done with ode45, and
% the errors, needed for the animation, are computed.

% Gains
K_t = 0.1;
K_rcm = 0.1 * eye(2);
K = [K_t 0 0; [0;0] K_rcm];

tspan = [0 35]; % time of integration
x0 = [q0; lambda0]; % robot initial state
dynamics = @(t, x) RR_kinematic_model(t, x, p_trocar, theta_d, K);
[t_out, x_out] = ode45(dynamics, tspan, x0);

% Results
q_history = x_out(:, 1:2);
lambda_history = x_out(:, 3);
rcm_error_history = zeros(length(t_out), 1);
theta_error_history = zeros(length(t_out), 1);

% Computing errors for the animation
for k = 1:length(t_out)
    q = q_history(k, :)';
    lambda = lambda_history(k);
    theta = sum(q);

    p1 = [cos(q(1)); sin(q(1))];
    p2 = [p1(1) + cos(q(1) + q(2)); p1(2) + sin(q(1) + q(2))];
    p_rcm = p1 + lambda * (p2 - p1);

    rcm_error_history(k) = norm(p_trocar - p_rcm);
    theta_error_history(k) = norm(theta_d - theta);
end

%% Animations
% Here the robot motion, the parameters evolution and the errors evolution
% graphs are shown. The first animation starts automatically, then there is
% a button at the bottom right to restart it.

f = figure('Position', [100 100 1300 800], 'Name', 'Animazione con Play');
uicontrol('Style', 'pushbutton', ...
    'String', '▶ Play', ...
    'FontSize', 12, ...
    'Position', [1180 20 100 40], ...
    'Callback', @(src,event) run_animation(f,t_out,q_history,lambda_history,theta_error_history,rcm_error_history,p_trocar,theta_d));

run_animation(f,t_out,q_history,lambda_history,theta_error_history,rcm_error_history,p_trocar,theta_d);

function run_animation(f,t_out,q_history,lambda_history,theta_error_history,rcm_error_history,p_trocar,theta_d)

for i = 1:2:length(q_history)
    q1 = q_history(i, 1);
    q2 = q_history(i, 2);
    lambda = lambda_history(i);
    p1 = [cos(q1); sin(q1)];
    p2 = [p1(1) + cos(q1 + q2); p1(2) + sin(q1 + q2)];
    p_rcm = p1 + lambda * (p2 - p1);

    delete(findall(f, 'Type', 'axes'))

    % --- Robot graph ---
    subplot('Position',[0.05, 0.55, 0.4, 0.4]);
    hold on; axis equal; grid on;
    xlim([-1 3]); ylim([-0.5 2]);
    
    % 2R robot's links
    plot([0 p1(1)], [0 p1(2)], 'b', 'LineWidth', 2, 'DisplayName','Link 1');
    plot([p1(1) p2(1)], [p1(2) p2(2)], 'r', 'LineWidth', 2, 'DisplayName','Link 2');

    % World Origin
    plot(0, 0, '^', 'MarkerSize', 10, 'MarkerEdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 2, 'DisplayName', 'World Origin');

    % Trocar and RCM points
    plot(p_trocar(1), p_trocar(2), 'bx', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Trocar');
    plot(p_rcm(1), p_rcm(2), 'o', 'MarkerSize', 10, 'MarkerEdgeColor', [1, 0.5, 0], 'LineWidth', 2, 'DisplayName', 'RCM');

    % Desired orientation
    dir_d = [cos(theta_d); sin(theta_d)];
    line_len = 1.5;
    plot([p_trocar(1), p_trocar(1) + line_len * dir_d(1)], ...
        [p_trocar(2), p_trocar(2) + line_len * dir_d(2)], 'k--', 'DisplayName', 'Desired Direction');

    legend('Location', 'northwest');
    title(sprintf('Time: %.2f s,  q_{1}= %.2f,  q_{2}= %.2f,  λ= %.2f', t_out(i), rad2deg(q1), rad2deg(q2), lambda));

    % --- Joint and lambda graphs ---
    subplot('Position',[0.48, 0.82, 0.45, 0.1]);
    plot(t_out(1:i), rad2deg(q_history(1:i,1)), 'g', 'LineWidth', 1.5);
    ylabel(sprintf('q_1'), 'Rotation', 0, 'FontWeight', 'bold');
    title(sprintf('λ: %.2fm,  q_{1}: %.2f°,  q_{2}: %.2f°',rad2deg(q1), rad2deg(q2), lambda));
    ylim([45 60]);
    set(gca, 'YAxisLocation', 'right');
    set(gca, 'XTickLabel', []);
    grid on;

    subplot('Position',[0.48, 0.7, 0.45, 0.1]);
    plot(t_out(1:i), rad2deg(q_history(1:i,2)), 'b', 'LineWidth', 1.5);
    ylabel(sprintf('q_{2}'), 'Rotation', 0, 'FontWeight', 'bold');
    ylim([-35 0]);
    set(gca, 'YAxisLocation', 'right');
    set(gca, 'XTickLabel', []);
    grid on;

    subplot('Position',[0.48, 0.58, 0.45, 0.1]);
    plot(t_out(1:i), lambda_history(1:i), 'r', 'LineWidth', 1.5);
    xlabel('Time (s)'); ylabel(sprintf('λ'), 'Rotation', 0, 'FontWeight', 'bold');
    ylim([0.4 0.6]);
    set(gca, 'YAxisLocation', 'right');
    grid on;

    % --- Error graphs ---
    subplot('Position',[0.05, 0.32, 0.9, 0.18]);
    plot(t_out(1:i), theta_error_history(1:i), 'LineWidth', 1.5);
    ylabel('Theta error (m)');
    title(sprintf('error_{theta}: %.2f,  error_{RCM}: %.2e',theta_error_history(i),rcm_error_history(i)));
    xlim([t_out(1) t_out(end)]);
    ylim([0 0.5]);
    set(gca, 'XTickLabel', []);
    grid on;

    subplot('Position',[0.05, 0.12, 0.9, 0.18]);
    plot(t_out(1:i), rcm_error_history(1:i), 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('RCM error (m)');
    xlim([t_out(1) t_out(end)]);
    ylim([-1e-6 2e-6]);
    grid on;

    drawnow;
end
end