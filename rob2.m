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
% clc
% syms a2 a3 q1 q2 q3 dc1 dc2 dc3 g0 real
% 
% table = [pi/2 0 0 q1;
%          0 a2 0 q2;
%          0 a3 0 q3];
% 
% rc1 = [0;dc1;0]; rc2 = [dc2-a2;0;0]; rc3 = [dc3-a3;0;0]; g_vector = [0;-g0;0];
% 
% 
% [T,M,c,g] = moving_frames(table,"RRR",[rc1;rc2;rc3],g_vector,false);

%% Midterm 13.04.2016
% ex 1
% clc
% syms q1 q2 q3 dq1 dq2 dq3 d2 d3 l2 I2 I3 m1 m2 m3 real
% T1 = 1/2 * m1*dq1;
% 
% pc2 = [q1+d2*cos(q2);
%        d2*sin(q2)];
% vc2 = diff_wrt(pc2,[q1,q2]);
% T2 = simplify(1/2 * m2 * norm(vc2)^2 + 1/2*I2*dq2^2);
% 
% pc3 = [q1+l2*cos(q2)+d3*cos(q2+q3);
%        l2*sin(q2)+d3*sin(q2+q3)];
% vc3 = diff_wrt(pc3,[q1,q2,q3]);
% T3 = simplify(1/2 * m3 * norm(vc3)^2 + 1/2*I3*(dq2+dq3)^2);
% 
% T = simplify(T1+T2+T3);
% M = simplify(hessian(T,[dq1,dq2,dq3]),'Steps', 20)
% c = coriolis_terms(M)

%% Midterm 13.04.2016
% ex 2
% clc
% syms q1 q2 q3 q4 l1 l2 l3 l4 d1 d2 d3 d4 g0 m1 m2 m3 m4 real
% 
% h1 = d1*sin(q1); h2 = l1*sin(q1)+d2*sin(q1+q2); h3 = l1*sin(q1)+l2*sin(q1+q2)+d3*sin(q1+q2+q3); h4 = l1*sin(q1)+l2*sin(q1+q2)+l3*sin(q1+q2+q3)+d4*sin(q1+q2+q3+q4);
% 
% U = simplify(m1*-g0*h1 + m2*-g0*h2 + m3*-g0*h3 + m4*-g0*h4);
% g = jacobian(U,[q1,q2,q3,q4]).';
% g = collect(collect(collect(collect(g,cos(q1)),cos(q1+q2)),cos(q1+q2+q3)),cos(q1+q2+q3+q4))

%% Midterm 13.04.2016
% ex 3
% clc
% syms l q1 q2 q3 q4 N real
% r = [l*(cos(q1)+cos(q1+q2)+cos(q1+q2+q3)+cos(q1+q2+q3+q4));
%      l*(sin(q1)+sin(q1+q2)+sin(q1+q2+q3)+sin(q1+q2+q3+q4));
%       q1 + q2 + q3 + q4];
% J = simplify(jacobian(r,[q1,q2,q3,q4]));
% % analysisJB(J)
% 
% dq0 = [0 0 1/64*pi/2 0]';
% 
% l = 0.5; q = [0 0 pi/2 0]'; v_d = [1 0]'; dtheta_d = 0.5;
% J_num = subs(J,[sym('l'),q1,q2,q3,q4],[l,q']);
% J_pinv_num = pinv(J_num);
% 
% q_command = double(-dq0 + J_pinv_num*([v_d;dtheta_d]+J_num*dq0));

%% Midterm 29.03.2017
% ex 1
% clc
% syms l1 l2 l3 q1 q2 q3 A F C D E g0 real
% 
% table = [pi/2 0 l1 q1;
%            0  l2 0 q2;
%            0  l3 0 q3];
% rc1 = [A -F 0]'; rc2 = [-C 0 0]'; rc3 = [-D 0 E]';
% [T,M,c] = moving_frames(table,"RRR",[rc1;rc2;rc3],[0;-g0;0],false)

%ex 3
% clc
% syms q1 q2 q3 q4 l real
% 
% table = [0 l 0 q1;
%          0 l 0 q2;
%          0 l 0 q3;
%          0 l 0 q4];
% 
% [~,r,~,~] = DK(table);
% J = simplify(jacobian(r(1:3,4),[q1,q2,q3,q4]));
% J = J(1:2,:);
% 
% J_num = double(subs(J,[l,q1,q2,q3,q4],[0.5,0,0,0,0]));
% q_try = pinv(J_num) * [0;10];
% 
% [dq, s, exitCode] = snsIk_vel_opt([-4;-2;-1;-1],[4;2;1;1],[0;10],J_num)

% clc
% syms q1 q2 q3  a2 a3 dc1 dc2 dc3 l2 real
% 
% table = [-pi/2 0 q1 0;
%          pi/2 0 l2 q2;
%          0    dc3 0 q3];
% 
% rc1 = [0;dc1;0]; rc2 = [0;l2-dc2;0]; rc3 = [0;0;0];
% 
% [T,M,~,~] = moving_frames(table,"PRR",[rc1;rc2;rc3],"diagonal",[0;-9;0],false)

% clc
% syms q1 q2 q3 q4 dq1 dq2 dq3 dq4 d1 d2 d3 d4 l1 l2 l3 l4 m1 m2 m3 m4 I3 I4 real
% 
% T1 = 0.5*m1*dq1^2;
% 
% T2 = 0.5*m2*(dq1^2+dq2^2);
% 
% T3 = 0.5* m3*(dq1^2+dq2^2+d3^2*dq3^2+2*d3*dq3*(dq2*cos(q3)-dq1*sin(q3))) + I3*dq3^2;
% 
% pc4 = [q1+l3*cos(q3)+d4*cos(q3+q4); q2+l3*sin(q3)+d4*sin(q3+q4)]; vc4 = diff_wrt(pc4,[q1,q2,q3,q4]);
% 
% T4 = 0.5*m4*(vc4.'*vc4) + 0.5*I4*(dq3+dq4)^2;
% 
% T_tot = T1+T2+T3+T4;
% 
% M = simplify(hessian(T_tot,[dq1,dq2,dq3,dq4]));

% syms a1 a2 a3 a4 a5 a6 real
% Ma = [a1 0 -a6*sin(q3)-a5*sin(q3+q4) -a5*sin(q3+q4);
%       0 a2 a6*cos(q3)+a5*cos(q3+q4) a5*cos(q3+q4);
%       -a6*sin(q3)-a5*sin(q3+q4) a6*sin(q3)+a5*sin(q3+q4) a3+2*a5*l3*cos(q4) a4+a5*l3*cos(q4);
%       -a5*sin(q3+q4) a5*cos(q3+q4) a4+a5*l3*cos(q4) a4];
% 
% c = coriolis_terms(Ma)

% clc
% syms m1 m2 l1 q1 l2 q2 g0 rc1x rc1y rc1z rc2x Ic1xx Ic1xy Ic1xz Ic1yy Ic1yz Ic1zz Ic2xx Ic2yy Ic2zz real
% 
% table = [-pi/2 0 l1 q1;
%              0 l2 0 q2];
% rc1 = [rc1x;rc1y;rc1z]; rc2 = [rc2x;0;0];
% Ic1 = [Ic1xx Ic1xy Ic1xz;
%        Ic1xy Ic1yy Ic1yz;
%        Ic1xz Ic1yz Ic1zz];
% Ic2 = [Ic2xx 0 0;
%        0 Ic2yy 0;
%        0 0 Ic2zz];
% [~, M, ~, g] = moving_frames(table,"RR",[rc1;rc2],{Ic1,Ic2},[-g0;0;0],false);
% % by inspection
% syms a1 a2 a3 a4 a5 a6 real
% c1 = Ic2xx + Ic1yy + m1*rc1x^2 + m1*rc1z^2;
% c2 = Ic2yy - Ic2xx + m2*(l2 + rc2x)^2;
% c3 = Ic2zz + m2*(l2 + rc2x)^2;
% M = rewrite(M);
% M=subs(M,[c1,c2,c3],[a1,a2,a3])
% c = coriolis_terms(M);
% expand(c)
% c5 = -m1*rc1x;
% c6 = -m1*rc1z;
% c4 = -m2*(l2 + rc2x);
% g = rewrite(g);
% g = subs(g,[c4,c5,c6],[a4,a5,a6])
% g = [a4*g0*cos(q1)*sin(q2) + a5*g0*sin(q1) + a6*g0*cos(q1);
%      a4*g0*cos(q1)*sin(q2)];
% [tau,Y,pi_coeffs] = regression_matrix(M,c,g,[a1;a2;a3;a4;a5;a6]);
% 
% syms t dq1 dq2 ddq1 ddq2 real
% 
% taud = subs(tau,[q1,q2,dq1,dq2,ddq1,ddq2],[2*t,pi/4,2,0,0,0])
% 
% eq1 = tau == 0;
% solve(eq1,[q1,q2])

% clc
% syms q1 q2 q3 dq1 dq2 dq3 real
% 
% V_max = [1.5;1.5;1]; A_max = [10;10;10];
% l=0.5;
% Tc=0.100;
% q = [0,0,pi/2]; dq = [0.8,0,-0.8];
% ddpd = [2 1]';
% table = [0 l 0 q1;
%          0 l 0 q2;
%          0 l 0 q3];
% 
% [p,T0N, R, A] = DK(table);
% r = T0N(1:2,4);
% J = simplify(jacobian(r,[q1,q2,q3]));
% dJ = simplify(diff_wrt(J,[q1,q2,q3]));
% 
% J = double(subs(J,[q1,q2,q3],q));
% dJ = double(subs(dJ,[q1,q2,q3,dq1,dq2,dq3],[q,dq]));
% dJdq=dJ*dq';
% 
% ddq = pinv(J)*(ddpd-dJdq)
% 
% Q_min = max(-((V_max+dq')/Tc),-A_max)
% Q_max = min(((V_max-dq')/Tc),A_max)
% 
% [ddq2, s, exitCode] = snsIk_acc_opt(Q_min,Q_max,ddpd,J,dJdq)

% clc
% syms q1 q2 q3 l1 l2 l3 rcx1 rcx2 rcx3 rcy1 rcy2 rcy3 g0 m2 m3 real
% 
% table = [0 l1 0 q1;
%          0 l2 0 q2;
%          0 l3 0 q3];
% 
% rc1 = [rcx1;rcy1;0]; rc2=[rcx2;rcy2;0]; rc3=[rcx3;rcy3;0];
% [T,M,c,g] = moving_frames(table,"RRR",[rc1;rc2;rc3],{},[g0;0;0],false)
% 
% g_new = subs(g,[m2,rcx1,rcx2,rcy2,rcx3,rcy3],[-m3,-l1,-l2,0,-l3,0])

% clc
% syms q1 q2 q3 q4 dq1 dq2 dq3 dq4 vxd vyd real
% 
% v1 = [dq1;0]; v2 = [dq1;dq2]; v3 = [dq1+dq3;dq2]; v4 = [dq1+dq3;dq2+dq4];
% 
% [T,M] = kinetic_energy("PPPP",{v1,v2,v3,v4},{})
% 
% J = jacobian([q1+q3;q2+q4],[q1,q2,q3,q4]);
% 
% %  building the inertia-weighted pseudoinverse
% Jm_pinv = simplify(M^-1*J'*(J*M^-1*J')^-1);
% vd = [vxd;vyd];
% dq_T = Jm_pinv * vd
% % minimizing the norm of q dot (granted by the pseudoinverse)
% dq_Q = pinv(J) * vd

% clc
% syms q1 q2 q3 l1 l3 rc1y rc2y rc3x real
% table = [pi/2 0 l1 q1;
%          -pi/2 0 q2 pi/2;
%          0 l3 0 q3];
% 
% rc1 = [0;rc1y;0]; rc2 = [0;rc2y;0]; rc3 = [rc3x;0;0];
% 
% [T,M,c,g] = moving_frames(table,"RPR",[rc1;rc2;rc3],{},[0;0;0],false);
% M_a = rewrite(M)
% 
% syms Ic3_xx Ic3_yy m3 Ic1_yy Ic2_xx m2 Ic3_zz real
% c1 = (Ic2_xx + Ic1_yy + Ic3_xx + m2*(q2 - rc2y)^2);
% c2 = -m2*(q2 - rc2y);
% c3 = m2 + m3;
% c4 = (-m3*(l3 + rc3x));
% c5 = -m3*(l3 + rc3x)^2;
% c6 = Ic3_zz + m3*(l3 + rc3x)^2;
% 
% syms a1 a2 a3 a4 a5 a6 real
% M_a = subs(M_a,[c1,c2,c3,c4,c5,c6],[a1,a2,a3,a4,a5,a6])

% clc
% syms a1 a2 a3 a4 a5 a6 q1 q2 q3 real
% 
% M = [a1+2*a2*q2+a3*q2^2+2*a4*q2*sin(q3)+a5*sin(q3)^2 0 0;
%      0 a3 a4*cos(q3);
%      0 a4*cos(q3) a6];
% a = [a1;a2;a3;a4;a5;a6];
% 
% c = coriolis_terms(M)
% [S1,S2,S3] = factorize_M(M);

%% Midterm 13.04.2022
% ex 1
% clc
% syms q1 q2 a1 a2 real
% 
% table = [0 a1 0 q1;
%          0 a2 0 q2];
% 
% q_a = [0,0]; p_a = [2,0,0];
% q_b = [pi/2,0]; p_b = [0,2,0];
% q_c = [pi/4,-pi/4]; p_c = [1.6925,0.7425,0];
% q_d = [0,pi/4]; p_d = [1.7218,0.6718,0];
% 
% l = calibration(table,["d"],[q_a;q_b;q_c;q_d],[p_a;p_b;p_c;p_d])

% ex 3
% clc
% syms q1 q2 q3 l1 l2 l3 dc1 dc2 dc3 real
% 
% table = [0 l1 0 q1;
%       pi/2 0  l2 q2;
%          0 l3 0 q3];
% rc1 = [-l1+dc1;0;0]; rc2 = [0;-l2+dc2;0]; rc3 = [-l3+dc3;0;0];
% 
% [T,M,c,g] = moving_frames(table,"RRR",[rc1;rc2;rc3],{},[0;0;0])

% ex 4
% clc
% syms q1 q2 q3 dq1 dq2 dq3 real
% 
% q = [q1;q2;q3]; dq = [dq1;dq2;dq3];
% q_hat = [2*pi/5;pi/2;-pi/4];
% 
% [~,T,~,~] = DK([0 1 0 q1;0 1 0 q2; 0 1 0 q3]);
% r=T(1:2,4);
% J = jacobian(r,q);
% J = double(subs(J,q,q_hat));
% J_pinv = pinv(J);
% 
% dr = [-3;0];
% 
% q_mid = [0; pi/3; 0];
% 
% q0 = 1/3 * [(q1/pi^2); (q2-pi/3)/(2*pi/3)^2; q3/(pi/2)^2];
% 
% dqn = subs(- (eye(3)-J_pinv*J)*q0,q,q_hat);
% dqr = subs(J_pinv*dr,q,q_hat);
% dq = dqr +dqn;
% dq_num = double(subs(dq,q,q_hat));
% 
% dq_num_scaled = double(((2-dqn(1))/dqr(1) * dqr) + dqn)
% 
% snsIk_vel_opt([-2;-2,;-2],[2;2;2],dr,J)

% ex 5
% clc
% syms a b c l2 q1 q2 dq1 dq2 ddq1 ddq2 rc1y dc2 m1 m2 g0 ddr real
% M = [a b*cos(q2); b*cos(q2) c];
% q=[q1,q2]; q_bar = [1,pi/2]; dq = [dq1;dq2];
% M_num = subs(M,q,q_bar);
% 
% r = q1+l2*sin(q2);
% 
% ddr_num = subs(ddr,q,q_bar);
% J = jacobian(r,q);
% dJ = diff_wrt(J,q,1);
% J_num = subs(J,q,q_bar);
% dJ_num = subs(dJ,q,q_bar);
% 
% J_pinv = pinv(J_num);
% 
% c = coriolis_terms(M);
% c_num = subs(c,q,q_bar);
% g = gravity_terms({m1,m2},[0;-g0;0],{[0;q1-rc1y;0],[dc2*cos(q2);dc2*sin(q2);0]});
% g_num = subs(g,q,q_bar);
% 
% tau_A = pinv(J_num*M_num^-1) * (ddr_num + dJ_num*dq + J_num*M_num^-1*(g_num));
% tau_B = (M_num * J_pinv) * (ddr_num + dJ_num*dq + J_num*M_num^-1*(g_num));
% 
% tau_A = simplify(tau_A)
% tau_B = simplify(tau_B)

% clc
% syms q1 q2 real
% syms a1 a2 a3 m1 m2 real positive
% 
% MA=[a1+a2*q1^2 a2; a2 a2];
% MB = [a1 a3*cos(q2-q1); a3*cos(q2-q1) a1];
% MC = [a1+2*a2*cos(q2) a1+a2*cos(q2); a1+a2*cos(q2) a1];
% MD = [m1+m2 -0.5*m2; -0.5*m2 m2];
% 
% is_inertia_matrix(MD)

% clc
% syms q1 q2 q3 dq1 dq2 dq3 m1 m2 m3 dc2 I2 l2 I3 g0 k1 real
% 
% pc2 = [dc2*cos(q2); q1+k1+dc2*sin(q2)];
% vc2 = diff_wrt(pc2,[q1,q2],1);
% pc3 = [l2*cos(q2)-q3*sin(q2); q1+k1+l2*sin(q2)+q3*cos(q2)];
% vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% 
% [T,M] = kinetic_energy("PRR", {dq1,vc2,vc3}, {0,dq2,dq2})
% 
% c = coriolis_terms(M)
% g = gravity_terms({m1,m2,m3},[0;-g0],{q1,pc2,pc3})
% 
% rewrite(M)
% 
% %by inspection 
% c1 = m1+m2+m3;
% c3 = m3;
% c4 = dc2*m2;
% c2 = I2 + I3 + m2*dc2^2;
% 
% syms a1 a2 a3 a4 real
% coeffs = [c1,c2,c3,c4];a = [a1,a2,a3,a4];
% M_a = subs(rewrite(M),c2,a2)
% M_a = subs(rewrite(M),coeffs,a)
% M_a(2,2) = a3*(l2^2+q3^2)+a2
% c_a = subs(rewrite(c),coeffs,a)
% 
% [tau,Y,pi_coeffs] = regression_matrix(M_a,c_a,0,a)
% 
% syms q1 dq1 ddq1 q2 dq2 ddq2 q3 dq3 ddq3
% joint_syms = [q1 dq1 ddq1 q2 dq2 ddq2 q3 dq3 ddq3];
% Y_m = subs(Y,[l2,dc2],[1,1]);
% is_minimal(Y_m,a,joint_syms)

% clc
% syms m1 m2 g0 dc1 dc2 l1 q1 q2 real
% 
% pc1 = [dc1*cos(q1); dc1*sin(q1)];
% pc2 = [l1*cos(q1)+dc2*cos(q2); l1*sin(q1)+dc2*sin(q2)];
% 
% g=gravity_terms({m1,m2},[g0;0],{pc1,pc2})

% clc
% syms a1 a2 a3 q2 q1 dq1 dq2 real
% M = [a1 a2*sin(q2); a2*sin(q2) a3];
% c = coriolis_terms(M);
% dq = [dq1;dq2];
% [S1,~,~] = factorize_M(M);
% 
% S2 = S1;
% S2(2,:) = [dq2 -dq1];
% clc
% c
% S1*dq
% S2*dq
% 
% dM = diff_wrt(M,[q1,q2],1);
% op3=simplify(dM - 2*S2)
% fprintf("skew_check: %s\n",is_skew(op3))

% clc
% syms a1 a2 a3 q1 q2 l1 l2 real
% M = [a1+2*a2*cos(q2) a3+a2*cos(q2);
%      a3+a2*cos(q2) a3];
% 
% table = [0 l1 0 q1; 0 l2 0 q2];
% [~,T0N,~,~] = DK(table);
% r = T0N(1:2,4);
% ro = norm(r);
% J = jacobian(ro,[q1,q2]);
% 
% J_M_pinv = simplify(M^-1*J.'*(J*M^-1*J.')^-1)
% J_M_pinv_num = double(subs(J_M_pinv,[q1,q2,l1,l2,a1,a2,a3],[0,pi/2,1,1,10,2.5,5/3]));
% J_M_pinv_num*0.5
% 
% clc
% syms q1 q2 q3 l dq1 dq2 dq3 real
% 
% table = [0 l 0 q1; 0 l 0 q2; 0 l 0 q3];
% 
% q = [pi/6;pi/6;pi/6]
% [~,T0N,~,~] = DK(table);
% r = T0N(1:2,4);
% J = jacobian(r,[q1,q2,q3]);
% J = double(subs(J,[q1,q2,q3,l],[q',1]));
% dJ = diff_wrt(J,[q1,q2,q3],1);
% dJdq = dJ * [dq1;dq2;dq3];
% [ddq, s, exitCode] = snsIk_acc_opt([-2.8;-3.6;-4],[2.8;3.6;4],[4;2],J,dJdq)