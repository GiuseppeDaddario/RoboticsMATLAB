function dx = RR_kinematic_model(~, x, p_trocar, theta_d, K)
% Control function for the 2R robot model. The state is given by joint angles q
% and lambda. Since the robot has unitary link lenght, l1 and l2 are not
% multiplied in the formulas

q = x(1:2);
lambda = x(3);

q1 = q(1); q2 = q(2);
theta = q1 + q2;
p1 = [cos(q1); sin(q1)];
p2 = p1 + [cos(q1 + q2); sin(q1 + q2)];
p_rcm = p1 + lambda * (p2 - p1);

% Jacobian
J1 = [-sin(q1) 0; cos(q1) 0];
J2 = [-sin(q1+q2)-sin(q1) -sin(q1+q2);
       cos(q1+q2)+cos(q1)  cos(q1+q2)];
J_rcm = [J1+lambda*(J2-J1) p2 - p1];
J = [1 1 0;
    J_rcm]; % final augmented jacobian

% Error
e = [theta_d - theta;
    p_trocar - p_rcm];

% Control: only the feedback part of the control scheme is used.
%  - the forward term is zero because the derivative of the task is zero;
%  - the projection in the null space is zero because the robot is not 
%    redundant for the task.
J_pinv = pinv(J);
dx = J_pinv * K * e;

% lambda interval
dx(3) = max(min(dx(3), 1), -1);
end