format short

%% Midterm 13.04.2022
% Exercise 1
% syms l1 l2 q1 q2
% assume([l1,l2,q1,q2],"real")
% r = [l1*cos(q1) + l2*cos(q1+q2);
%      l1*sin(q1) + l2*sin(q1+q2)];
% 
% qa = [0;0]; qb = [pi/2;0]; qc= [pi/4;-pi/4]; qd = [0;pi/4];
% 
% % data from experiments
% pa_e = [2;0]; pb_e = [0;2]; pc_e = [1.6925;0.7425]; pd_e = [1.7218;0.6718]; 
% 
% % nominal data
% pa_n = subs(r,[l1,l2,q1,q2],[1,1,qa(1),qa(2)]);
% pb_n = subs(r,[l1,l2,q1,q2],[1,1,qb(1),qb(2)]);
% pc_n = double(subs(r,[l1,l2,q1,q2],[1,1,qc(1),qc(2)]));
% pd_n = double(subs(r,[l1,l2,q1,q2],[1,1,qd(1),qd(2)]));
% 
% % equation
% delta_r_bar = [pa_e-pa_n; pb_e-pb_n; pc_e-pc_n; pd_e-pd_n];
% delta_r_bar = double(delta_r_bar)
% 
% omega = jacobian(r,[l1,l2]);
% omega_bar = [subs(omega,[q1,q2],[qa(1),qa(2)]);
%              subs(omega,[q1,q2],[qb(1),qb(2)]);
%              subs(omega,[q1,q2],[qc(1),qc(2)]);
%              subs(omega,[q1,q2],[qd(1),qd(2)])];
% omega_bar = double(omega_bar)
% 
% % result
% delta_ro = pinv(omega_bar)*delta_r_bar;
% delta_ro = double(delta_ro)
% 
% l_real = delta_ro + 1 

%% Midterm 29.03.2017
% Exercise 5
% syms l1 l2 q1 q2
% f = [l1*cos(q1) + l2*cos(q1+q2);
%      l2*sin(q1) + l2*sin(q1+q2)];
% 
% simplify(jacobian(f,[q1,q2]))

%% Midterm 24.04
% Exercise 1
% syms q1 q2 q3 real
% C = [0;2]; v = [0;1];
% q = [0;pi/2;-pi/2];
% table = [0 1 0 q1;
%          0 1 0 q2;
%          0 1 0 q3];
% pe = DK(table); pe = pe(1:2);
% pm = DK(table(1:2,:)); pm = pm(1:2);
% Je = jacobian(pe,[q1,q2,q3]);
% Jm = jacobian(pm,[q1,q2,q3]);
% 
% pm_num = subs(pm,[q1,q2,q3],q');
% pe_num = subs(pe,[q1,q2,q3],q');
% Jm_num = subs(Jm,[q1,q2,q3],q');
% Je_num = subs(Je,[q1,q2,q3],q');
% Je_pinv = pinv(Je_num);
% 
% q0 = double(1/2*Jm_num'* ((pm_num-C)/norm(pm_num-C))); %traspose because of the dimensions, but check if is the same after
% q_pg = double(q0 + Je_pinv*(v - Je_num*q0))
% vm = double(Jm_num*q_pg)
% 
% %TP
% Pe = eye(3)-Je_pinv*Je_num;
% q_tp = double(Je_pinv*v + pinv(Jm_num*Pe)*(vm-Jm_num*Je_pinv*v))

%% Midterm 19.04
% Exercise 1
% clc
% syms q1 q2 q3 q1_d q2_d q3_d real
% table = [0 0.5 0 q1;
%          0 0.5 0 q2;
%          0 0.5 0 q3];
% fk = DK(table); fk = fk(1:2)
% q = [q1;q2;q3]; q_d = [q1_d;q2_d;q3_d];
% J_sim = jacobian(fk,q);
% J_dot_sim = diff(J_sim) * q_d
% 
% q0 = [0;0;pi/2];
% q_dot = [0.8 0 -0.8]';
% a = [2;1];
% 
% J = double(subs(J_sim,q,q0))
% J_dot = double(subs(J_dot_sim,q,q0))
% 
% q_ddot = pinv(J)*(a-J_dot*q_dot)

%% Exam 05.06 2020
% Exercise 1
% clc
% V1=1; V2=1.5; V3=2;
% V_limits = [V1;V2;V3];
% J = [-1 -1 -0.5; -0.366 -0.866 -0.866];
% v = [2;1];
% 
% [dq, s, exitCode] = snsIk_vel_opt(-V_limits,V_limits,v,J)

%% Exam 12.06.2024
% exercise 1
clc
syms l1 l2 q1 q2 d1 d2 g0 real

table = [0 l1 0 q1;
         0 l2 0 q2];

% table = [-pi/2 0 l1 q1;
%          0 l2 0 q2];
rc1 = [-l1+d1;0;0]; rc2 = [-l2+d2;0;0]; g_vector = [0;-g0;0];
% rc1_ = [rc1x;rc1y;rc1z]; rc2_ = [rc2x;0;0];
% 
[T,M,c,g] = moving_frames(table,"RR",[rc1;rc2],g_vector,false);
S = factorize_M(M);

