clear; clc; close all;

%% Initial configuration
% Here there are the definitions of the robot parameters (q1,q2,q3) in the initial
% configuration and the desired orientation. Since the robot has unitary
% link lenght, l1 and l2 are not multiplied in the formulas

q0 = [0; pi/4; 0];
lambda = 0.5;
theta_d = pi/8;

p0 = [q0(1); 0];
p1 = [p0(1) + cos(q0(2)); p0(2) + sin(q0(2))];
p2 = [p1(1) + cos(q0(2) + q0(3)); p1(2) + sin(q0(2) + q0(3))];
p_trocar = p1 + lambda * (p2 - p1);

%% Integration
% Here the gains are set; The integration, done with ode45, and
% the errors, needed for the animation, are computed.
% For this task both the trocar-rcm error and the final orientation error
% are in the same order of magnitude of the first task, but with a lower value. 
% Also in this task I've tried to bound the trocar-rcm error's oscillations
% as much as possible, and the result was better than in the first task.

% Gains
K_t = 1;
K_rcm = 1 * eye(2);
K_gains = [K_t 0 0; [0;0] K_rcm];

tspan = linspace(0,5,1000);
x0 = q0; % robot initial state
dynamics = @(t, x) PRR_kinematic_model(t, x, p_trocar, theta_d, K_gains, lambda);
[t_out, x_out] = ode45(dynamics, tspan, x0);

% Results
q_history = x_out; % q_hist now has 3 columns: q1_p, q2_r, q3_r
rcm_error_history = zeros(length(t_out), 1);
theta_error_history = zeros(length(t_out), 1);

% Computing errors for the animation
for k = 1:length(t_out)
    q_current = q_history(k, :)';
    q1 = q_current(1);
    q2 = q_current(2);
    q3 = q_current(3);
    theta = q2+q3;

    p0_prime = [q1; 0];
    p1 = [p0_prime(1) + cos(q2); p0_prime(2) + sin(q2)];
    p2 = [p1(1) + cos(q2 + q3); p1(2) + sin(q2 + q3)];
    p_rcm_current = p1 + lambda * (p2 - p1);

    rcm_error_history(k) = norm(p_trocar - p_rcm_current);
    theta_error_history(k) = norm(theta_d-theta);
end

%% Animation and graphs
% Here the robot motion, the parameters evolution and the errors evolution
% graphs are shown. The first animation starts automatically, then there is
% a button at the bottom right to restart it.

f = figure('Position', [100 100 1300 800], 'Name', 'PPR robot animation');
uicontrol('Style', 'pushbutton', ...
    'String', '▶ Play', ...
    'FontSize', 12, ...
    'Position', [1180 20 100 40], ...
    'Callback', @(src,event) run_animation(f,t_out,q_history,lambda,theta_error_history,rcm_error_history,p_trocar,theta_d));

run_animation(f,t_out,q_history,lambda,theta_error_history,rcm_error_history,p_trocar,theta_d);

function run_animation(f,t_out,q_history,lambda,theta_error_history,rcm_error_history,p_trocar,theta_d)

% Precomputations
n = length(q_history);
idx_step = 14;
indices = 1:idx_step:n;

% resetting the graphs
delete(findall(f, 'Type', 'axes'));

% --- Robot graph ---
ax1 = subplot('Position',[0.05, 0.55, 0.4, 0.4], 'Parent', f);
hold(ax1, 'on'); axis(ax1, 'equal'); grid(ax1, 'on');
xlim(ax1, [-0.5 3]); ylim(ax1, [-0.15 1.75]);

link1 = plot(ax1, [0 0], [0 0], 'Color', [0, 0.7, 0], 'LineWidth', 2,'DisplayName','Link 1');
link2 = plot(ax1, [0 0], [0 0], 'b', 'LineWidth', 2,'DisplayName','Link 2');
link3 = plot(ax1, [0 0], [0 0], 'Color', [1, 0.2, 0.2], 'LineWidth', 2,'DisplayName','Link 3');
plot(ax1, 0, 0, '^', 'MarkerSize', 10, 'MarkerEdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 2,'DisplayName','Robot base');
trocar_plot = plot(ax1, p_trocar(1), p_trocar(2), 'bx', 'MarkerSize', 12, 'LineWidth', 2,'DisplayName','Trocar point');
rcm_plot = plot(ax1, 0, 0, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', [1, 0.5, 0], 'LineWidth', 2,'DisplayName','Rcm point');
dir_d = [cos(theta_d); sin(theta_d)];
dir_plot = plot(ax1, [0 0], [0 0], 'k--','DisplayName','Desired orientation');

legend(ax1, 'Location', 'northwest');

% --- Joint graphs ---
ax_q1 = subplot('Position', [0.48, 0.82, 0.45, 0.1], 'Parent', f);
q1_plot = plot(ax_q1, t_out(1), q_history(1,1), 'g', 'LineWidth', 1.5);
ylim(ax_q1, [0 0.11]);
xlim(ax_q1, [t_out(1), t_out(end)]);
set(ax_q1, 'XTickLabel', [], 'YAxisLocation', 'left');
grid(ax_q1, 'on');
yl_1 = ylabel(ax_q1, 'q_1', 'Rotation', 0);
set(yl_1, 'Units', 'normalized', 'Position', [1.03, 0.4, 0]);

ax_q2 = subplot('Position',[0.48, 0.7, 0.45, 0.1], 'Parent', f);
q2_plot = plot(ax_q2, t_out(1), rad2deg(q_history(1,2)), 'b', 'LineWidth', 1.5);
ylim(ax_q2, [45 61]);
xlim(ax_q2, [t_out(1), t_out(end)]);
set(ax_q2, 'XTickLabel', [],'YAxisLocation','left');
grid(ax_q2, 'on');
yl_2 = ylabel(ax_q2, sprintf('q_{2}'),'Rotation',0); 
set(yl_2, 'Units', 'normalized', 'Position', [1.03, 0.4, 0]);

ax_q3 = subplot('Position',[0.48, 0.58, 0.45, 0.1], 'Parent', f);
q3_plot = plot(ax_q3, t_out(1), rad2deg(q_history(1,3)), 'r', 'LineWidth', 1.5); 
xlabel(ax_q3, 'Time (s)');
xlim(ax_q3, [t_out(1), t_out(end)]); 
ylim(ax_q3, [-40 0]); 
grid(ax_q3, 'on');
yl_3 = ylabel(ax_q3, sprintf('q_{3}'),'Rotation',0);
set(yl_3, 'Units', 'normalized', 'Position', [1.03, 0.4, 0]);

% --- Error graphs ---
ax_theta_err = subplot('Position',[0.05, 0.32, 0.9, 0.18], 'Parent', f);
theta_plot = plot(ax_theta_err, t_out(1), theta_error_history(1), 'LineWidth', 1.5);
ylabel(ax_theta_err, 'Theta error (m)');
xlim(ax_theta_err, [t_out(1) t_out(end)]); ylim(ax_theta_err, [0 0.5]); set(ax_theta_err, 'XTickLabel', []); grid(ax_theta_err, 'on');

ax_rcm_err = subplot('Position',[0.05, 0.12, 0.9, 0.18], 'Parent', f);
rcm_plot_err = plot(ax_rcm_err, t_out(1), rcm_error_history(1), 'LineWidth', 1.5);
ylabel(ax_rcm_err, 'RCM error (m)'); xlabel(ax_rcm_err, 'Time (s)');
xlim(ax_rcm_err, [t_out(1) t_out(end)]); ylim(ax_rcm_err, [-0.4e-6 2.7e-6]); grid(ax_rcm_err, 'on');

for i = indices
    q = q_history(i, :)';
    q1 = q(1); q2 = q(2); q3 = q(3);

    p0 = [q1; 0];
    p1 = p0 + [cos(q2); sin(q2)];
    p2 = p1 + [cos(q2 + q3); sin(q2 + q3)];
    p_rcm = p1 + lambda * (p2 - p1);

    % Update robot plot
    set(link1, 'XData', [0 p0(1)], 'YData', [0 p0(2)]);
    set(link2, 'XData', [p0(1) p1(1)], 'YData', [p0(2) p1(2)]);
    set(link3, 'XData', [p1(1) p2(1)], 'YData', [p1(2) p2(2)]);
    set(rcm_plot, 'XData', p_rcm(1), 'YData', p_rcm(2));
    set(dir_plot, 'XData', [p_trocar(1), p_trocar(1) + 1.5 * dir_d(1)], ...
        'YData', [p_trocar(2), p_trocar(2) + 1.5 * dir_d(2)]);

    title(ax1, sprintf('Time: %.2f s,  q_{1}= %.2f,  q_{2}= %.2f°,  q_{3}= %.2f°,  λ= %.2f', ...
        t_out(i), q1, rad2deg(q2), rad2deg(q3), lambda));

    % Update joint plots
    set(q1_plot, 'XData', t_out(1:i), 'YData', q_history(1:i,1));
    set(q2_plot, 'XData', t_out(1:i), 'YData', rad2deg(q_history(1:i,2)));
    set(q3_plot, 'XData', t_out(1:i), 'YData', rad2deg(q_history(1:i,3)));

    % Update error plots
    set(theta_plot, 'XData', t_out(1:i), 'YData', theta_error_history(1:i));
    set(rcm_plot_err, 'XData', t_out(1:i), 'YData', rcm_error_history(1:i));

    title(ax_theta_err, sprintf('error_{theta}: %.2e,  error_{RCM}: %.2e', ...
        theta_error_history(i), rcm_error_history(i)));

    drawnow;
end
end
