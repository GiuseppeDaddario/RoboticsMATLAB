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
% Here the gains are set; The integration, done with ode45, and
% the errors, needed for the animation, are computed.
% For this task I've tried to bound the trocar-rcm error's oscillations as
% much as possible, 'fine-tuning' the trade-off between gains and integration time/steps,
% slightly compromising the accuracy of the final orientation (which is still in the order of 10^-3)

% Gains
K_t = 1;
K_rcm = 1 * eye(2);
K = [K_t 0 0; [0;0] K_rcm];

tspan = linspace(0,4.5,1000); % time of integration
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

f = figure('Position', [100 100 1300 800], 'Name', '2R robot animation');
uicontrol('Style', 'pushbutton', ...
    'String', '▶ Play', ...
    'FontSize', 12, ...
    'Position', [1180 20 100 40], ...
    'Callback', @(src,event) run_animation(f,t_out,q_history,lambda_history,theta_error_history,rcm_error_history,p_trocar,theta_d));

run_animation(f,t_out,q_history,lambda_history,theta_error_history,rcm_error_history,p_trocar,theta_d);

function run_animation(f,t_out,q_history,lambda_history,theta_error_history,rcm_error_history,p_trocar,theta_d)

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

link1 = plot(ax1, [0 0], [0 0], 'b', 'LineWidth', 2,'DisplayName','Link 1');
link2 = plot(ax1, [0 0], [0 0], 'r', 'LineWidth', 2,'DisplayName','Link 2');
plot(ax1, 0, 0, '^', 'MarkerSize', 10, 'MarkerEdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 2,'DisplayName','Robot base');
trocar_plot = plot(ax1, p_trocar(1), p_trocar(2), 'bx', 'MarkerSize', 12, 'LineWidth', 2,'DisplayName','Trocar point');
rcm_plot = plot(ax1, 0, 0, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', [1, 0.5, 0], 'LineWidth', 2,'DisplayName','Rcm point');
dir_d = [cos(theta_d); sin(theta_d)];
dir_plot = plot(ax1, [0 0], [0 0], 'k--','DisplayName','Desired orientation');

legend(ax1, 'Location', 'northwest');

% --- Joint and lambda graphs ---
ax_q1 = subplot('Position',[0.48, 0.82, 0.45, 0.1], 'Parent', f);
q1_plot = plot(ax_q1, t_out(1), rad2deg(q_history(1,1)), 'g', 'LineWidth', 1.5);
ylim(ax_q1, [45 58]); 
set(ax_q1, 'XTickLabel', []); 
grid(ax_q1, 'on');
xlim(ax_q1, [t_out(1), t_out(end)]);
yl_1 = ylabel(ax_q1, sprintf('q_{1}'), 'Rotation', 0);
set(yl_1, 'Units', 'normalized', 'Position', [1.03, 0.4, 0]);

ax_q2 = subplot('Position',[0.48, 0.7, 0.45, 0.1], 'Parent', f);
q2_plot = plot(ax_q2, t_out(1), rad2deg(q_history(1,2)), 'b', 'LineWidth', 1.5);
ylim(ax_q2, [-35 0]); 
set(ax_q2, 'XTickLabel', []); 
grid(ax_q2, 'on');
xlim(ax_q2, [t_out(1), t_out(end)]);
yl_2 = ylabel(ax_q2, sprintf('q_{2}'), 'Rotation', 0);
set(yl_2, 'Units', 'normalized', 'Position', [1.03, 0.4, 0]);

ax_lambda = subplot('Position',[0.48, 0.58, 0.45, 0.1], 'Parent', f);
lambda_plot = plot(ax_lambda, t_out(1), lambda_history(1), 'r', 'LineWidth', 1.5);
ylim(ax_lambda, [0.50 0.57]); 
xlabel(ax_lambda, 'Time (s)'); 
grid(ax_lambda, 'on');
xlim(ax_lambda, [t_out(1), t_out(end)]);
yl_lambda = ylabel(ax_lambda, 'λ', 'Rotation', 0);
set(yl_lambda, 'Units', 'normalized', 'Position', [1.03, 0.4, 0]);

% --- Error graphs ---
ax_theta_err = subplot('Position',[0.05, 0.32, 0.9, 0.18], 'Parent', f);
theta_plot = plot(ax_theta_err, t_out(1), theta_error_history(1), 'LineWidth', 1.5);
ylabel(ax_theta_err, 'Theta error (m)');
xlim(ax_theta_err, [t_out(1) t_out(end)]); ylim(ax_theta_err, [0 0.5]); set(ax_theta_err, 'XTickLabel', []); grid(ax_theta_err, 'on');

ax_rcm_err = subplot('Position',[0.05, 0.12, 0.9, 0.18], 'Parent', f);
rcm_plot_err = plot(ax_rcm_err, t_out(1), rcm_error_history(1), 'LineWidth', 1.5);
ylabel(ax_rcm_err, 'RCM error (m)'); xlabel(ax_rcm_err, 'Time (s)');
xlim(ax_rcm_err, [t_out(1) t_out(end)]); ylim(ax_rcm_err, [-0.2e-6 5e-6]); grid(ax_rcm_err, 'on');

for i = indices
    q1 = q_history(i,1);
    q2 = q_history(i,2);
    lambda = lambda_history(i);

    p1 = [cos(q1); sin(q1)];
    p2 = p1 + [cos(q1+q2); sin(q1+q2)];
    p_rcm = p1 + lambda * (p2 - p1);

    % Update robot plot
    set(link1, 'XData', [0 p1(1)], 'YData', [0 p1(2)]);
    set(link2, 'XData', [p1(1) p2(1)], 'YData', [p1(2) p2(2)]);
    set(rcm_plot, 'XData', p_rcm(1), 'YData', p_rcm(2));
    set(dir_plot, 'XData', [p_trocar(1), p_trocar(1) + 1.5 * dir_d(1)], ...
        'YData', [p_trocar(2), p_trocar(2) + 1.5 * dir_d(2)]);

    title(ax1, sprintf('Time: %.2f s,  q_1= %.2f°,  q_2= %.2f°,  λ= %.2f', ...
        t_out(i), rad2deg(q1), rad2deg(q2), lambda));

    % Update joint and lambda plots
    set(q1_plot, 'XData', t_out(1:i), 'YData', rad2deg(q_history(1:i,1)));
    title(ax_q1, sprintf('λ: %.2fm,  q_1: %.2f°,  q_2: %.2f°', ...
        lambda, rad2deg(q1), rad2deg(q2)));

    set(q2_plot, 'XData', t_out(1:i), 'YData', rad2deg(q_history(1:i,2)));
    set(lambda_plot, 'XData', t_out(1:i), 'YData', lambda_history(1:i));

    % Update error plots
    set(theta_plot, 'XData', t_out(1:i), 'YData', theta_error_history(1:i));
    set(rcm_plot_err, 'XData', t_out(1:i), 'YData', rcm_error_history(1:i));

    title(ax_theta_err, sprintf('error_θ: %.2e,  error_{RCM}: %.2e', ...
        theta_error_history(i), rcm_error_history(i)));

    drawnow;
end
end