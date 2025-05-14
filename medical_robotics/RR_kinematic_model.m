function dxdt = RR_kinematic_model(~, x, p_trocar, theta_d, K)
% Control function for the 2R robot model. The state is given by joint angles q
% and lambda. Since the robot has unitary link lenght, l1 and l2 are not in the formulas

% Robot state
q = x(1:2); q1 = q(1); q2 = q(2);
lambda = x(3);

% Direct Kinematics
c1 = cos(q1);
s1 = sin(q1);
c12 = cos(q1 + q2);
s12 = sin(q1 + q2);

p1 = [c1; s1];
p2 = p1 + [c12; s12];
p_rcm = p1 + lambda * (p2 - p1);
theta = q1 + q2;

% Jacobian
J1 = [-s1 0; c1 0];
J2 = [-s12-s1 -s12;
       c12+c1  c12];
J_rcm = [J1+lambda*(J2-J1) p2 - p1];
% final augmented jacobian
J = [1 1 0; % task jacobian
    J_rcm]; 

% Error
e = [theta_d - theta;
    p_trocar - p_rcm];

% Control: only the feedback part of the control scheme is used.
%  - the forward term is zero because the derivative of the task is zero;
%  - the projection in the null space is zero because the robot is not 
%    redundant for the task.
J_pinv = pinv(J);
dxdt = J_pinv * K * e;

% lambda interval
dxdt(3) = max(min(dxdt(3), 1), -1);
end