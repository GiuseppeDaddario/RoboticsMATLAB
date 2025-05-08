format short;
% ex3
% clc
% syms m1 m2 m3 q1 q2 q3 l1 l2 l3 g0 dc1 dc2 dc3 real
% 
% table = [-pi/2 l1 q1 0;
%             0 l2 q2 pi/2;
%             0 l3 0 q3];
% 
% rc1 = [-l1+dc1;-q1;0];  rc2 = [-l2+dc2; l1;q2]; rc3 = [l1-l3+dc3;-q1+l2;0];
% syms I3xx I3xy I3xz I3yy I3yz I3zz real
% I3 = [I3xx I3xy I3xz; I3xy I3yy I3yz; I3xz I3yz I3zz];
% [T,M,c,g] = moving_frames(table,"PPR",[rc1;rc2;rc3],{eye(3),eye(3),I3},[0;0;-g0])
% [S1,S2,S3] = factorize_M(M)
% M_re = rewrite(M)
% syms a1 a2 a3 a4 real
% c1 = m1+m2+m3;
% c2 = m3*(dc3+l1);
% c3 = m2+m3;
% c4 = I3zz + m3*(dc3 + l1)^2;
% coeffs = [c1,c2,c3,c4]; a = [a1,a2,a3,a4];
% M_re = subs(M_re,coeffs,a)
% M_re(3,3) = a4;
% c_re = subs(c,coeffs,a)
% g_re = subs(g,coeffs,a)
% [tau,Y,pi_coeffs] = regression_matrix(M_re,c_re,g_re,[a1,a2,a3,a4])
% is_minimal(Y,a,[q1 dq1 ddq1 ddq2 q2 dq2 ddq2 q3 dq3 ddq3 g0])
% 
% syms w t
% M_num = subs(M_re,q3,(w*t))
% c_num = subs(c_re,q3,(w*t))
% g_num = subs(g_re,q3,(w*t))
% 
% simplify(M_num*[ddq1;ddq2;ddq3]+c_num+g_num)
%ex 2

% syms l1 l2 l3 q1 q2 q3 dc1 dc2 dc3 ddq1 ddq2 ddq3 real
% table = [pi/2 0 l1 q1; 0 l2 0 q2; 0 l3 0 q3];
% rc1 = [0;-l1+dc1;0]; rc2 = [-l2+dc2;0;0]; rc3 = [-l3+dc3; 0;0];
% [T,M,c,g] = moving_frames(table,"RRR",[rc1;rc2;rc3],{},[0;0;0])
% 
% dM = diff_wrt(M,[q1,q2,q3],1);
% ddq = [ddq1;ddq2;ddq3];
% nabla_H = simplify(ddq.'*M*dM*ddq)
% 
% [~,T0N,~,~] = DK(table);
% r = T0N(1:3,4);
% J = jacobian(r,[q1,q2,q3]);
% dJ = diff_wrt(J,[q1,q2,q3],1);
% 
% q_val = [3*pi/2 pi/4 0];
% J_num = subs(J,[q1,q2,q3,l1,l2,l3],[q_val,0.5,0.5,0.5])
% dJ_num = subs(dJ,[q1,q2,q3,l1,l2,l3,dq1,dq2,dq3],[q_val,0.5,0.5,0.5,0,0,0])
% 
% J_pinv = pinv(J_num);
% nabla_H_num = vpa(subs(nabla_H,[q1,q2,q3,l1,l2,l3,dq1,dq2,dq3,m1,m2,m3,dc1,dc2,dc3],[q_val,0.5,0.5,0.5,0,0,0,2,2,2,0.25,0.25,0.25]),2)
% ddq_eq = J_pinv * ([-1;-1;0]-dJ_num*[dq1;dq2;dq3]) + (eye(3)-J_pinv*J_num)*nabla_H_num



%ex 3
clc

syms q1 q2 q3 q4 q5 real
q = [0; pi/3; -pi/4;pi/2;-pi/3];
dr1 = [2;3]; dr2 =[2;-0.5]; dr3 = [-1;0];

table = [0 1 0 q1; 0 1 0 q2; 0 1 0 q3; 0 1 0 q4; 0 1 0 q5];

[~,T0N,~,~] = DK(table);
r = T0N(1:3,4);
r1 = [cos(q1)+cos(q2)+cos(q3)+cos(q4)+cos(q5); sin(q1)+sin(q2)+sin(q3)+sin(q4)+sin(q5)];
r2 = [cos(q1)+cos(q2)+cos(q3); sin(q1)+sin(q2)+sin(q3)];
r3 = [cos(q1)+cos(q2); sin(q1)+sin(q2)];

% 
% r2 = [cos(q1 + q2 + q3)+cos(q1 + q2) + cos(q1);sin(q1 + q2 + q3)+sin(q1 + q2) + sin(q1)];
% r3 = [cos(q1 + q2) + cos(q1);sin(q1 + q2) + sin(q1)];

J1 = jacobian(r1,[q1,q2,q3,q4,q5]);
J2 = jacobian(r2,[q1,q2,q3]);
J3 = jacobian(r3,[q1,q2]);

J1_num = double(subs(J1,[q1,q2,q3,q4,q5],q'));
J2_num = double(subs(J2,[q1,q2,q3],[q(1),q(2),q(3)]));
J2_num = [J2_num [0;0] [0;0]; [0 0 0 0 0]];
J3_num = double(subs(J3,[q1,q2],[q(1),q(2)]));
J3_num = [J3_num [0;0] [0;0] [0;0]; [0 0 0 0 0]];


PA1 = eye(5) - (pinv(J1_num)*J1_num);
PA2 = eye(5) - (pinv(J2_num)*J2_num);


dq1 = pinv(J1_num)*dr1;
dq2 = dq1 + pinv(J2_num*PA1) * (dr2-J2_num*dq1);
d_tp = dq2 + pinv(J3_num*PA2) * (dr3-J3_num*dq2)

e1 = norm(dr1-(J1_num*d_tp))
e2 = norm(dr2-(J2_num*d_tp))
e3 = norm(dr3-(J3_num*d_tp))

% dr_A = [dr1;dr2;dr3];
% J_A = [J1_num;J2_num;J3_num];
% q_Ta = pinv(J_A)*dr_A
% 
% e1_A = norm(dr1-(J1_num*q_Ta))
% e2_A = norm(dr2-(J2_num*q_Ta))
% e3_A = norm(dr3-(J3_num*q_Ta))
% 
% J2 = [J2 [0;0] [0;0]; [0 0 0 0 0]];
% J3= [J3 [0;0] [0;0] [0;0]; [0 0 0 0 0]];
% 
% eq1 = norm(dr1-(J1*q_Ta)) == 0;
% eq2 = norm(dr2-(J2*q_Ta)) == 0;
% eq3 = norm(dr3-(J3*q_Ta)) == 0; 
% 
% eq1 = subs(eq1,[q4,q5],[pi/2,-pi/3])
% eq2 = subs(eq2,[q4,q5],[pi/2,-pi/3])
% eq3 = subs(eq3,[q4,q5],[pi/2,-pi/3])
% 
% solve([eq1,eq2,eq3],[q1,q2],"ReturnConditions",true)
