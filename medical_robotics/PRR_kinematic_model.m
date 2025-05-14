function dxdt = PRR_kinematic_model(~, x, p_trocar, theta_d, K_gains, lambda)
% Control function for the PRR robot model. The state is given by joint angles q
% Since the robot has unitary revolute link lenght, l2 and l3 are not in the formulas

% Robot state
q1 = x(1);
q2 = x(2);
q3 = x(3);

% Direct Kinematics
c2 = cos(q2);
s2 = sin(q2);
c23 = cos(q2 + q3);
s23 = sin(q2 + q3);

p1 = [q1; 0];
p2 = [p1(1) + c2;
             s2];
p3 = [p2(1) + c23;
      p2(2) + s23];
p_rcm = p2 + lambda * (p3 - p2);
theta = q2 + q3;

% Jacobian
% J_rcm has been computed after manually differentition of the RCM point 
J_rcm = [1 -s2-lambda*s23 -lambda*s23; 
         0  c2+lambda*c23  lambda*c23];
% final augmented jacobian
J = [0 1 1; % task jacobian
    J_rcm];

% Error
e = [theta_d - theta; p_trocar - p_rcm];

% Control: only the feedback part of the control scheme is used.
%  - the forward term is zero because the time derivative of the task is zero;
%  - the projection in the null space is zero because the robot is not
%    redundant for the task, since the degree of fredom introduced is used
%    to keep lambda fixed.
dxdt = pinv(J) * K_gains * e;
end