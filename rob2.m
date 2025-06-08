format short

%% Exam 19.09.2024 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_24.09.19.pdf]
%ex 1
% clc
% syms q1 q2 dc1 m1 m2 I1 I2 l1 dq1 dq2 g0 real
% 
% pc1 = [dc1*cos(q1); dc1*sin(q1)];
% vc1 = diff_wrt(pc1,q1,1);
% 
% pc2 = [l1*cos(q1) + q2*sin(q1); l1*sin(q1)-q2*cos(q1)];
% vc2 = diff_wrt(pc2,[q1,q2],1);
% 
% T1 = 0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2;
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*dq1^2;
% [T_tot,M] = kinetic_energy({T1,T2},[dq1,dq2])
% c = coriolis_terms(M);
% g = gravity_terms({m1,m2},[0;-g0],{pc1,pc2});
% 
% %%
% syms a1 a2 a3 a4 ddq1 ddq2 real
% c1 = I1 + I2 + m1*dc1^2 + m2*q2^2;
% c2 = m2;
% c3 = dc1*m1 + l1*m2;
% 
% M_a = [a1+a2*l1^2 -a2*l1; -a2*l1 a2];
% c_a = [2*dq1*dq2*a2*q2; -dq1^2*a2*q2];
% g_a = [g0*a2*q2*sin(q1) + g0*a3*cos(q1); (-g0*a2)*cos(q1)];
% 
% [tau,Y,pi_coeffs] = regression_matrix(M_a,c_a,g_a,[a1,a2,a3])
% is_minimal(Y,[a1,a2,a3],[q1,q2,dq1,dq2])

%ex2 
% clc
% syms a b k T real positive
% syms t real
% syms q2 m1 dc1 m2 I1 I2 l1 real
% q = [a+b*(1-cos(pi*t/T)); k];
% dq = diff(q,t);
% M = [m1*dc1^2 + m2*l1^2 + m2*q2^2 + I1 + I2, -l1*m2;
%                                      -l1*m2,     m2];
% M=subs(M,q2,q(2));
% int_p=int(M*dq,t,0,T);

%ex3
% clc
% syms q2 m1 dc1 m2 I1 I2 l1 real
% Ma = [m1*dc1^2 + m2*l1^2 + m2*q2^2 + I1 + I2, -l1*m2];
% Mb = [-l1*m2, m2];
% M = [Ma;
%      Mb];
% 
% pinv_Ma_M = M^-1*Ma.'*(Ma*M^-1*Ma.')^-1;
% pinv_Ma_M = simplify(pinv_Ma_M*c)

%ex4
%TODO

%ex5
% clc
% syms m1 dc1 m2 I1 I2 l1 t T a b k q1 q2 dq1 dq2 ddq1 ddq2 a1 real
% q = [a+b*(1-cos((pi*t)/T)); k];
% dq = [pi/T*b*sin((pi*t)/T);0];
% ddq = [(pi/T)^2*b*cos((pi*t)/T);0];
% 
% 
% M = [m1*dc1^2 + m2*l1^2 + m2*q2^2 + I1 + I2, -l1*m2;
%                                     -l1*m2,     m2];
% c = coriolis_terms(M);
% sub = [q(1), q(2), dq(1), dq(2), ddq(1), ddq(2)];
% 
% 
% tau = simplify(M*ddq+c);
% tau = subs(tau, [m2*l1^2 + m1*dc1^2 + I1 + I2,q1,q2,dq1,dq2,ddq1,ddq2],[a1,sub])
% 
% syms tau_max_1 real
% T_star = sqrt((a1+m2*k^2)*b*pi^2/tau_max_1);
% tau_2_t = simplify(subs(tau(2),t,0));
% tau_2_t = simplify(subs(tau_2_t,T,T_star))
% tau_2_T = simplify(subs(tau(2),t,T));
% tau_2_T = simplify(subs(tau_2_T,T,T_star))

%% Exam 12.06.2024
% ex1
clear; clc
syms q1 q2 q3 a2 a3 dc1 dc2 dc3 real
table = [pi/2 0 0 q1;
           0 a2 0 q2;
           0 a3 0 q3];

[p, T0N, R, A] = DK(table);

% a2=1;a3=1;
% plot_robot(A, [q1,q2,q3], [1,pi/2,0]);

rc1 = [0;dc1;0]; rc2 = [dc2-a2;0;0]; rc3 = [dc3-a3;0;0];
[T,M,c,g] = moving_frames(table,"RRR",[rc1;rc2;rc3],[],[0;0;0])

% %%
% syms ro1 ro2 ro3 ro4 ro5 ro6 Ic3_zz m3 real
% 
% c3 = Ic2_yy + m2*dc2^2 + m3*a2^2
% c4 = Ic2_zz + Ic3_zz + m3*a2^2 + m2*dc2^2 + m3*dc3^2;
% c5 = a2*dc3*m3;
% c6 = Ic3_zz + m3*dc3^2;

%%
% [c, t] = coeffs(M(1,1), [cos(q2+q3)]);
% idx = find(t == cos(q2+q3));
% coeff_x2 = c(idx);
% raw2latex(coeff_x2)

% ex2
% clc
% syms q1 q2 q3 q4 real
% 
% table = [0 1 0 q1; 0 1 0 q2; 0 1 0 q3; 0 1 0 q4];
% [r,~,~,A] = DK(table);clc
% 
% r = r(1:2);
% J = jacobian(r,[q1,q2,q3,q4]);
% 
% 
% q1_m = -pi/2; q1_M = pi/2; q1_mid = (q1_M+q1_M)/2;
% q2_m = 0; q2_M = pi/2; q2_mid = (q2_M+q2_M)/2;
% q3_m = -pi/4; q3_M = pi/4; q3_mid = (q3_M+q3_M)/2;
% q4_m = -pi/4; q4_M = pi/4; q4_mid = (q4_M+q4_M)/2;
% 
% dq0 = 1/4 * [(q1-q1_mid)/(q1_M-q1_m)^2;
%              (q2-q2_mid)/(q2_M-q2_m)^2;
%              (q3-q3_mid)/(q3_M-q3_m)^2;
%              (q4-q4_mid)/(q4_M-q4_m)^2];
% %% numerical
% dpt = [-1;-1];
% %plot_robot(A, [q1,q2,q3,q4], [0,pi/2,0,-pi/4])
% q1 = 0; q2 = pi/2; q3 = 0; q4 = -pi/4;
% J_num = eval(J);
% J_pinv = pinv(J_num);
% dq0_num = eval(dq0);
% 
% dq_num = double(dq0_num + J_pinv*(dpt - J_num*dq0_num));
% 
%% Exam 12.06.23 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_23.06.12.pdf]
% ex1
% clc
% syms q1 q2 q3 q4 dq1 dq2 dq3 dq4 dc1 dc2 dc3 dc4 l1 l2 l3 l4 m1 m2 m3 m4 I1 I2 I3 I4 real
% 
% pc1 = [dc1*cos(q1); 
%        dc1*sin(q1)]; 
% vc1 = diff_wrt(pc1,q1,1);
% 
% pc2 = [l1*cos(q1)+dc2*cos(q2); 
%        l1*sin(q1)+dc2*sin(q2)]; 
% vc2 = diff_wrt(pc2,[q1,q2],1);
% 
% pc3 = [l1*cos(q1)+l2*cos(q2)+dc3*cos(q3);
%        l1*sin(q1)+l2*sin(q2)+dc3*sin(q3)];
% vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% 
% pc4 = [l1*cos(q1)+l2*cos(q2)+l3*cos(q3)+dc4*cos(q4);
%        l1*sin(q1)+l2*sin(q2)+l3*sin(q3)+dc4*sin(q4)];
% vc4 = diff_wrt(pc4,[q1,q2,q3,q4],1);
% 
% T1 = 0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2;
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*dq2^2;
% T3 = 0.5*m3*(vc3.'*vc3) + 0.5*I3*dq3^2;
% T4 = 0.5*m4*(vc4.'*vc4) + 0.5*I4*dq4^2;
% 
% [T,M] = kinetic_energy({T1,T2,T3,T4},[dq1,dq2,dq3,dq4]);
% 
% %%
% A = [1 0 0 0;
%      1 1 0 0;
%      1 1 1 0;
%      1 1 1 1];
% 
% M_theta = simplify(A.' * M * A);
% 
% syms m11 m12 m13 m14 m22 m23 m24 m33 m34 m44 real
% M_theta_sub = simplify(subs(M_theta,[M(1,1),M(1,2),M(1,3),M(1,3),M(2,1),M(2,2),M(2,3),M(2,4),M(3,1),M(3,2),M(3,3),M(3,4),M(4,1),M(4,2),M(4,3),M(4,4)], ...
%     [m11,m12,m13,m14,m12,m22,m23,m24,m13,m23,m33,m34,m14,m24,m34,m44]),'Steps',100);

%ex2 
% clc
% syms q1 q2 q3 q4 real
% table = [0 1 0 q1;
%          0 1 0 q2;
%          0 1 0 q3;
%          0 1 0 q4];
% 
% % [~,~,~,A] = DK(table);clc
% % plot_robot(A,[q1,q2,q3,q4],q0)
% 
% r = [cos(q1)+cos(q2)+cos(q3)+cos(q4); sin(q1)+sin(q2)+sin(q3)+sin(q4)];
% q0 = [0, pi/6, -pi/3, -pi/3]; 
% 
% J = simplify(jacobian(r,[q1,q2,q3,q4]));
% J_e = double(subs(J,[q1,q2,q3,q4],q0));
% J_pinv = pinv(J_e);
% J2 = simplify(jacobian(r,[q1,q2]));
% J2 = [J2 [0;0] [0;0]];
% J_t = double(subs(J2,[q1,q2,q3,q4],q0));
% J_t_pinv = pinv(J_t);
% 
% %% a - b: pseudoinverse
% clc
% ve = [0.4330;-0.75];
% vt = [-0.5;0.8660];
% 
% dq_e = J_pinv*ve
% dq_t = J_t_pinv*vt
% 
% a_error_e = ve - J_e*dq_e
% a_error_e_norm = norm(a_error_e)
% a_error_t = vt - J_t*dq_e
% a_error_t_norm = norm(a_error_t)
% 
% b_error_e = ve - J_e*dq_t
% b_error_e_norm = norm(b_error_e)
% b_error_t = vt - J_t*dq_t
% b_error_t_norm = norm(b_error_t)
% 
% %% c: task augmentation
% clc
% JA = [J_e; J_t];
% 
% dq_A = pinv(JA)*[ve;vt]
% 
% c_error = [ve;vt] - JA*dq_A
% c_error_norm = norm(c_error)
% 
% %% d - e: task priority
% clc
% Pe = eye(4)-(J_pinv*J_e);
% dq_TP_e = dq_e + pinv(J_t*Pe)*(vt-J_t*dq_e)
% Pt = eye(4)-(J_t_pinv*J_t);
% dq_TP_t = dq_t + pinv(J_e*Pt)*(ve-J_e*dq_t)
% 
% d_error_e = ve - J_e*dq_TP_e
% d_error_e_norm = norm(d_error_e)
% d_error_t = vt - J_t*dq_TP_e
% d_error_t_norm = norm(d_error_t)
% 
% e_error_e = ve - J_e*dq_TP_t
% e_error_e_norm = norm(e_error_e)
% e_error_t = vt - J_t*dq_TP_t
% e_error_t_norm = norm(e_error_t)

%ex3
% clc
% syms m1 m2 m3 g0 q1 q2 dc1 dc2 l2 real
% 
% rc1 = [0;q1-dc1;0]; 
% rc2 = [dc2*cos(q2); q1+dc2*sin(q2);0]; 
% rc3 = [l2*cos(q2); q1+l2*sin(q2);0];
% 
% g = gravity_terms([m1,m2,m3],[0;-g0;0],{rc1,rc2,rc3})

%ex4
% clc
% syms a1 a2 a3 q1 q2 m1 m2 mp g0 real
% 
% M=[a1+2*a2*cos(q2) a3+a2*cos(q2); a3+a2*cos(q2) a3];
% 
% c = coriolis_terms(M);
% 
% pc1 = [0.5*cos(q1); 0.5*sin(q1);0];
% pc2 = [cos(q1)+0.5*cos(q1+q2); sin(q1)+0.5*sin(q1+q2);0];
% pcp = [cos(q1)+cos(q1+q2); sin(q1)+sin(q1+q2);0];
% g = gravity_terms([m1,m2,mp],[0;-g0;0],{pc1,pc2,pcp});
% g = g(1:2);
% %%
% syms a4 a5 real
% c4 = (m1/2 + m2 + mp)*g0;
% c5 = g0*(m2/2 + mp);
% g = [a4*cos(q1) + a5*cos(q1+q2); a5*cos(q1+q2)];
% %%
% h = cos(q1)+cos(q1+q2);
% [dyn,lambda,A,D,E,F] = reduced_dynamics(M,c,g,h)

%% Exam 13.02.2023 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_23.02.13.pdf]
% ex1
% clc
% syms l m q1 q2 q3 dq1 dq2 dq3 real
% Ic = (m*l^2)/12;
% q_bar = [pi/4,-pi/2,pi/2];
% ddp_d = [1;0];
% 
% %computing kinetic energies
% pc1 = [l/2 * cos(q1); l/2 * sin(q1)]; vc1 = diff_wrt(pc1,q1,1);
% pc2 = [l*cos(q1)+l/2*cos(q1+q2); l*sin(q1)+l/2*sin(q1+q2)]; vc2 = diff_wrt(pc2,[q1,q2],1);
% pc3 = [l*cos(q1)+l*cos(q1+q2)+l/2*cos(q1+q2+q3); l*sin(q1)+l*sin(q1+q2)+l/2*sin(q1+q2+q3)]; vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% 
% p = [l*cos(q1)+l*cos(q1+q2)+l*cos(q1+q2+q3); l*sin(q1)+l*sin(q1+q2)+l*sin(q1+q2+q3)]; dp = diff_wrt(p,[q1,q2,q3],1);
% J = jacobian(p,[q1,q2,q3]);
% 
% T1 = 0.5*m*(vc1.'*vc1) + 0.5*Ic*dq1^2;
% T2 = 0.5*m*(vc2.'*vc2) + 0.5*Ic*(dq1+dq2)^2;
% T3 = 0.5*m*(vc3.'*vc3) + 0.5*Ic*(dq1+dq2+dq3)^2;
% 
% [T,M] = kinetic_energy({T1,T2,T3},[dq1,dq2,dq3]);
% 
% c=coriolis_terms(M);
% 
% %% eval
% dq1=0; dq2=0; dq3=0;
% %c_bar=eval(c); % is 0
% M_bar=eval(M);
% Jpinv = pinv(J);
% ddq = simplify(Jpinv*ddp_d);
% tau = simplify(M_bar*ddq);
% q1=q_bar(1); q2=q_bar(2); q3=q_bar(3);
% tau_A = vpa(tau,4)
% %%
% T = [1 0 0; 1 1 0; 1 1 1]; W = T.'*T;
% J=eval(J);
% M_bar=eval(M);
% Jpinv_W = W^-1*J.'*(J*W^-1*J.')^-1;
% ddq_B = simplify(Jpinv_W*ddp_d);
% tau_B = simplify(M_bar*ddq_B);
% tau_B = vpa(tau_B,4)
% %%
% M_bar=eval(M);
% W = M_bar;
% J=eval(J);
% Jpinv_M = W^-1*J.'*(J*W^-1*J.')^-1;
% ddq_C = simplify(Jpinv_M*ddp_d);
% tau_C = simplify(M_bar*ddq_C);
% tau_C = vpa(tau_C,4)


%% ex2
% clc
% syms t tau T real
% 
% [t1,t1v,t1a] = cubic("cubicTime",[0,pi,0,0],T)
% I = 1.5;
% mgd = 17.715;
% umax=20;
% Tmin = sqrt((6*pi*I)/umax)
% t1 = subs(t1,tau,t/T)
% pltPVA({t1,t1v,t1a},0,Tmin)

%% ex 3
% clc
% syms m1 m2 q1 q2 dq1 dq2 ddq1 ddq2 K Kp1 Kp2 Kd1 Kd2 q1d q2d real
% V = 0.5*m1*dq1^2 + 0.5*m2*dq2^2 + 0.5*K*(q1-q2)^2 + 0.5*Kp1*(q1d-q1)^2 + 0.5*Kp2*(q2d-q2)^2;
% dV = m1*dq1*ddq1 + m2*dq2*ddq2 + K*(q1-q2)*(dq1-dq2) - Kp1*(q1d-q1)*dq1 - Kp2*(q2d-q2)*dq2
% %%
% F1 = Kp1*(q1d-q1) - Kd1*dq1; F2 = Kp2*(q2d-q2) - Kd2*dq2;
% ddq1 = (F1-K*(q1-q2))/m1; ddq2 = (F2-K*(q2-q1))/m2;
% dV = simplify(eval(dV))
% %%
% K_bar = [K+Kp1 -K; -K K+Kp2];
% qd_bar = [Kp1*q1d;Kp2*q2d];
% q_bar = simplify(K_bar^-1*qd_bar,'Steps',500)


%% Exam 25.01.2023 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_23.01.25.pdf]
%ex1 
% clc
% syms q1 q2 dq1 dq2 m1 m2 I1 I2 g0 l1 l2 dc1 dc2 real
% 
% rc1 = [dc1*cos(q1); dc1*sin(q1);0]; vc1 = diff_wrt(rc1,q1,1);
% rc2 = [l1*cos(q1); l1*sin(q1);0]; vc2 = diff_wrt(rc2,q1,1);
% 
% T1 = simplify(0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2);
% T2 = simplify(0.5*m2*(vc2.'*vc2) + 0.5*I2*(dq1+dq2)^2);
% 
% [T,M] = kinetic_energy({T1,T2},[dq1,dq2])
% %c = coriolis_terms(M) %is zero
% g = gravity_terms([m1,m2],[g0;0;0],{rc1,rc2});
% 
% syms I_tot real
% M_sub = subs(M,m1*dc1^2 + m2*l1^2 + I1 + I2,I_tot);
% 
% syms ddq1 ddq2 Fv1 Fv2 real
% ddq = [ddq1;ddq2]; dq = [dq1;dq2];
% Fv = [Fv1,0;
%      0,Fv2]; 
% 
% c2 = I2;
% c1= I_tot;
% c3 = dc1*m1 + l1*m2;
% c4 = Fv1;
% c5 = Fv2;
% 
% syms a1 a2 a3 a4 a5 real
% M_sub = subs(M_sub,[c1,c2,c3,c4,c5],[a1,a2,a3,a4,a5]);
% g = subs(g,[c1,c2,c3,c4,c5],[a1,a2,a3,a4,a5]);
% Fv = subs(Fv,[c4,c5],[a4,a5]);
% tau = M_sub*ddq + g + Fv*dq
% 
% Y = jacobian(tau,[a1,a2,a3,a4,a5])
% %%
% ddif = diff(g,q1)

%ex2 
% clc
% syms m1 m2 q1 q2 dq1 dq2 dc1 dc2 f_n tau1 tau2 ddq1 ddq2 real
% pc1 = [q1-dc1; 0]; vc1 = [dq1;0];
% pc2 = [q1;q2-dc2]; vc2 = [dq1;dq2];
% T1 = 0.5*m1*dq1^2;
% T2 = 0.5*m2*(vc2.'*vc2);
% [T,M] = kinetic_energy({T1,T2},[dq1,dq2]);
% R = getRfromEulerAngles("Z"); R = R(1:2,1:2); %dal nuovo frame a quello vecchio
% v_t = R.'*[dq1;dq2];
% tau_bar = R.'*[tau1;tau2];
% M_bar = R.'*M;
% dv = R.'*[ddq1;ddq2];
% f = [0;f_n];
% dynamic_model = M_bar*dv == tau_bar+f;
% raw2latex(simplify(dynamic_model))



%% Exam 21.10.2022 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_22.10.21.pdf]
% ex1
% clc
% syms m1 m2 q1 q2 dq1 dq2 dc2 Ic2 g0 real
% 
% rc1 = [q1;0;0]; vc1 = [dq1;0;0];
% rc2 = [q1+dc2*cos(q2); dc2*sin(q2);0]; vc2 = diff_wrt(rc2,[q1,q2],1);
% T1 = 0.5*m1*(vc1.'*vc1);
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*Ic2*dq2^2;
% [T,M] = kinetic_energy({T1,T2},[dq1,dq2]);
% c = coriolis_terms(M);
% g = gravity_terms([m1,m2],[g0;0;0],{rc1,rc2});
% 
% %% 2
% syms a1 a2 a3 ddq1 ddq2 real
% c1 = m1+m2;
% c2 = Ic2 + m2*dc2^2;
% c3 = dc2*m2;
% 
% M_a = subs(M,[c1,c2,c3],[a1,a2,a3]);
% c_a = subs(c,[c1,c2,c3],[a1,a2,a3]);
% g_a = subs(g,[c1,c2,c3],[a1,a2,a3]);
% 
% ddq = [ddq1;ddq2];
% tau = simplify(M_a*ddq + c_a + g_a);
% Y = jacobian(tau,[a1,a2,a3])
% %% 3
% syms u l2 q2d Kp1 Kp2 Kd1 Kd2 dqd1 dqd2 real
% Kp = [Kp1 0; 0 Kp2]; Kd = [Kd1 0; 0 Kd2];
% qd = [0;pi]; dqd = [dqd1; dqd2]; dq = [dq1; dq2];
% g_hat = -g0*[m1+m2; -l2*m2*sin(q2d)];
% u = Kp*(qd-[q1;q2]) + Kd*(dqd-dq) + g_hat
% %% 4
% q_0 = [0;0];
% dq_0 = [0;0];
% u_4 = subs(u,[q1,q2,dq1,dq2],[q_0',dq_0'])

%% Exam 09.09.2022 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_22.09.09.pdf]
% ex1
% clc
% syms q1 q2 q3 dq1 dq2 dq3 L real
% 
% q_max = (3*pi)/4; q_min = -(3*pi)/4;
% q_mid = (q_min+q_max)/2;
% 
% %coeff=(1/(12*(3*pi/4)^2)); % 0.0150
% dq0 = 0.0150 * [q1;q2;q3];
% 
% p = [L*(cos(q1)+cos(q1+q2)+cos(q1+q2+q3)); 
%      L*(sin(q1)+sin(q1+q2)+sin(q1+q2+q3))];
% 
% J = jacobian(p,[q1,q2,q3]);
% J_pinv = pinv(J);
% 
% dq = simplify(dq0 - J_pinv*J*dq0,'Steps',500)
% 
% %%
% q1 = 0; q2 = 2*pi/3; q3=q2;
% eval(dq)
% eval(dq0)

%% ex2
% clc
% syms q1 q2 dq1 dq2 m2 I1 I2 dc2 pd dpd ddpd real
% T1 = 0.5*I1*dq1^2;
% pc2 = [(q2-dc2)*cos(q1);
%        (q2-dc2)*sin(q1)];
% vc2 = diff_wrt(pc2,[q1,q2],1);
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*dq1^2;
% [T,M] = kinetic_energy({T1,T2},[dq1,dq2]);
% M = simplify(M,'Steps',500);
% c = coriolis_terms(M);
% 
% p = [q2*cos(q1);
%      q2*sin(q1)];
% J = jacobian(p,[q1,q2]);
% J_inv = J^-1;
% dJ = diff_wrt(J,[q1,q2],1);
% 
% P_0 = subs(p,[q1,q2],[0,2]); %initial error
% P_i = [2;3]; P_f = [-2;0];
% T = 2; V = 3;
% 
% beta = atan2(3/5,4/5);
% alpha = beta - pi;
% R = getRfromEulerAngles("Z",alpha); R = R(1:2,1:2);
% Kp = [4 0; 0 16]; Kd = [4 0; 0 8];
% 
% e = pd-p; de = dpd - diff_wrt(p,[q1,q2],1);
% 
% 
% 
% %% t = 0
% m2 = 10; dc2 = 2.5; I1 = 20; I2=20;
% ddpd = R.'*(9*((P_f-P_i)/5)); % A of bang-coast-bang
% dpd = 0;
% pd = P_i;
% e = R.'*[0;3];
% q1 = 0; q2 = 2;
% dq1 = 0; dq2 = 0;
% %M = eval(M); J=eval(J); J_inv = eval(J_inv); c = eval(c); dJ= eval(dJ);
% 
% a = ddpd +(Kp*e);
% u = M*J_inv*a + c - M*J_inv*dJ*[dq1;dq2]

%% Exam 08.07.2022 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_22.07.08.pdf]
% ex1
% clc
% syms m11 m22 m23 m33 q1 q2 q3 p1 p2 p3 c1 c2 c3 real
% M = [m11 0 0; 0 m22 m23; 0 m23 m33];
% T = [1 0 0; 0 1 0; 0 1 1];
% M_bar = simplify((T^-1).'*M*T^-1);
% c = [c1;c2;c3];
% c_bar = simplify((T^-1).'*c);
% 
% raw2latex(M_bar)
% raw2latex(c_bar)

% ex2
% clc
% syms m11 m12 m13 m14 m15 m22 m23 m24 m25 m33 m34 m35 m44 m45 m55 real
% F = [0 0 0 0;
%      1 0 0 0;
%      0 1 0 0;
%      0 0 1 0;
%      0 0 0 1];
% E = [1;
%      0;
%      0;
%      0;
%      0];
% M = [m11 m12 m13 m14 m15;
%      m12 m22 m23 m24 m25;
%      m13 m23 m33 m34 m35;
%      m14 m24 m34 m44 m45;
%      m15 m25 m35 m45 m55];
% syms tau1 tau2 tau3 tau4 tau5 n1 n2 n3 n4 n5 real
% tau = [tau1;tau2;tau3;tau4;tau5];
% n = [n1;n2;n3;n4;n5];
% simplify(E.'*tau)

% ex3
% clc
% syms m1 m2 q1 q2 dq1 dq2 dc1 dc2 g0 real
% pc1 = [q1-dc1;0;0]; vc1 = [dq1;0];
% pc2 = [q1;q2-dc2;0]; vc2 = [dq1; dq2];
% p = [q1;q2];
% J = jacobian(p,[q1,q2]);
% T1 = 0.5*m1*dq1^2; T2 = 0.5*m2*(vc2.'*vc2);
% [T,M] = kinetic_energy({T1,T2},[dq1,dq2])
% g = gravity_terms([m1,m2],[0;-g0;0],{pc1,pc2})
% %%
% syms U_max X Y real
% tx = 2*sqrt(X*(m1+m2)/U_max);
% ty = sqrt((4*Y*U_max*m2)/(U_max^2-(m2*g0)^2));
% 
% %%
% Ps = [1;0.3]; Pg = [0.6;0.7]; m1 = 5; m2 = 3; U_max = 40; g0 = 9.81;
% X = abs(Ps(1)-Pg(1)); Y = abs(Ps(2)-Pg(2));
% tx_num = eval(tx); ty_num=eval(ty);
% t = max(tx_num,ty_num)

%% Exam 10.06.2022 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_22.06.10.pdf]
% ex1
% clc
% syms A B C q1 q2 dq1 dq2 dq1_r dq2_r ddq1_r ddq2_r real
% M = [A B*cos(q2); B*cos(q2) C];
% ddqr = [ddq1_r;ddq2_r]; dqr = [dq1_r;dq2_r];
% c = coriolis_terms(M);
% [S,~,~] = factorize_M(M);
% tau = simplify(M * ddqr + S*dqr);
% Y = simplify(jacobian(tau, [A,B,C]))

% ex2
% clc
% syms q1 q2 q3 q4 L real
% table = [0 L 0 q1;
%          0 L 0 q2;
%          0 L/4 0 q3;
%          0 L/4 0 q4];
% [r,~,~,~] = DK(table)
% J = jacobian(r,[q1,q2,q3,q4]);
% J = J(1:2,:);
% 
% [r2,~,~,~] = DK(table(1:2,:));
% J2 = jacobian(r2,[q1,q2]);
% J2 = J2(1:2,:);
% J2 = [J2 zeros(2)];
% 
% J_ext = simplify([J;J2])

%% Exam 03.02.2022 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_22.02.03.pdf]
% ex1
% clc
% syms q1 q2 q3 dq1 dq2 dq3 dc1 dc2 dc3 l1 l3 g0 m1 m2 m3 I1 I2 I3 real
% 
% pc1 = [dc1*cos(q1);
%        dc1*sin(q1)];
% vc1 = diff_wrt(pc1,q1,1);
% 
% T1 = 0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2;
% 
% pc2 = [l1*cos(q1)+(q2-dc2)*cos(q1+pi/2);
%        l1*sin(q1)+(q2-dc2)*cos(q1)];
% vc2 = diff_wrt(pc2,[q1,q2],1);
% 
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*dq1^2;
% 
% pc3 = [l1*cos(q1)+q2*cos(q1-pi/2)+dc3*cos(q1+q3);
%        l1*sin(q1)+q2*sin(q1+pi/2)+dc3*sin(q1+q3)];
% vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% 
% T3 = 0.5*m3*(vc3.'*vc3) + 0.5*I3*(dq1+dq3)^2;
% 
% [T,M] = kinetic_energy({T1,T2,T3},[dq1,dq2,dq3]);
% 
% c = coriolis_terms(M);
% 
% g = gravity_terms([m1,m2,m3],[0;-g0;0],{[pc1;0],[pc2;0],[pc3;0]});
% raw2latex(rewrite(g))
% %%
% syms a1 a2 a3 a4 real
% a = [a1,a2,a3,a4];
% cs = [m2+m3, m1*dc1, m2*dc2, m3*dc3];
% g_sub = subs(rewrite(g),cs,a);
% g_sub(1) = cos(q1)*(a2*g0+g0*l1*a1)-sin(q1)*(g0*(a1*q2-a3)) + a4*g0*cos(q1+q3);
% G = simplify(jacobian(g_sub,a))
% %%
% syms l3 real
% r = [l1*cos(q1)+q2*cos(q1+pi/2)+l3*cos(q1+q3);
%        l1*sin(q1)+q2*sin(q1+pi/2)+l3*sin(q1+q3)];
% J = simplify(jacobian(r,[q1,q2,q3]))
% %%
% syms kp1 kp2 ep1 ep2 real
% Kp = [kp1 0; 0 kp2]; ep = [ep1;ep2];
% simplify(J.'*Kp*ep)

% ex2
% clc
% syms q1 q2 q3 q4 l real
% table = [0 l 0 q1;
%          0 l 0 q2;
%          0 l 0 q3;
%          0 l 0 q4];
% 
% [r,T0N, R, A] = DK(table); clc
% r = r(1:2,:);
% 
% J = simplify(jacobian(r,[q1,q2,q3,q4]));
% 
% Jr = [-sin(q1) -sin(q1+q2) -sin(q1+q2+q3) -sin(q1+q2+q3+q4);
%        cos(q1)  cos(q1+q2)  cos(q1+q2+q3)  cos(q1+q2+q3+q4)];
% 
% T = [1 0 0 0; 1 1 0 0; 1 1 1 0; 1 1 1 1];
% simplify(det(Jr*Jr.'),'Steps',1000)
% %%
% J_pinv = pinv(J);
% dJ = diff_wrt(J,[q1,q2,q3,q4],1);
% ddr = [1;1]; dq = [-0.8; 1; 0.2; 0]; Kd = eye(4);
% ddx = [ddr;0] - dJ*dq;
% q1 = pi/4; q2 = pi/3; q3 = -pi/2; q4 = 0;
% %%
% syms dq1 dq2 dq3 dq4 real
% dq1 = dq(1); dq2 = dq(2); dq3 = dq(3); dq4 = dq(4);
% ddqa = J_pinv * ddx - (eye(4)-J_pinv*J)* Kd*dq;
% l=1;
% eval(ddqa)
% %%
% syms q1 q2 q3 q4 real
% Jr = [-sin(q1) -sin(q1+q2) -sin(q1+q2+q3) -sin(q1+q2+q3+q4);
%        cos(q1)  cos(q1+q2)  cos(q1+q2+q3)  cos(q1+q2+q3+q4)];
% T = [1 0 0 0; 1 1 0 0; 1 1 1 0; 1 1 1 1];
% J = [Jr*T; 1 1 1 1];
% %%
% table = [0 l 0 q1;
%          0 l 0 q2;
%          0 l 0 q3;
%          0 l 0 q4];
% 
% [r,T0N, R, A] = DK(table); clc
% 
% q1 = pi/4; q2 = pi/3; q3 = -pi/2; q4 = 0;
% 
% r_ev = [eval(r(1:2)); q1+q2+q3+q4]
% eval(J*dq)

% ex3
% clc
% syms q1 q2 dc1 dc2 dq1 dq2 m1 m2 I1 I2 g0 real
% A = [-sin(q1) -sin(q1+q2)];
% D = [1 0];
% k = [A;D].';
% E = k(:,1); F = k(:,2);
% 
% pc1 = [dc1*cos(q1); dc1*sin(q1)]; vc1 = diff_wrt(pc1,q1,1);
% pc2 = [cos(q1)+dc2*cos(q1+q2); sin(q1)+dc2*sin(q1+q2)]; vc2 = diff_wrt(pc2,[q1,q2],1);
% 
% T1 = 0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2;
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*(dq1+dq2)^2;
% 
% [T,M] = kinetic_energy({T1,T2},[dq1,dq2]);
% c = coriolis_terms(M);
% g = gravity_terms([m1,m2],[0;-g0;0],{[pc1;0],[pc2;0]});
% %%
% syms a1 a2 a3 a4 a5 real
% c1 = I2 + m2*dc2^2 + I1 + m1*dc1^2 + m2;
% c2 = dc2*m2;
% c3 = I2 + + m2*dc2^2;
% c4 = g0*m2 + g0*dc1*m1;
% c5 = g0*dc2*m2;
% 
% M = [a1+(2*a2)*cos(q2) a3+a2*cos(q2); a3+a2*cos(q2) a3];
% c = [(-a2*dq2*(2*dq1 + dq2))*sin(q2); a2*dq1^2*m2*sin(q2)];
% g = [a4*cos(q1)+a5*cos(q1+q2); a5*cos(q1+q2)];
% 
% %%
% syms ddq1 u real
% dA = diff_wrt(A,[q1,q2],1);
% reduced_model = (F.'*M*F)*ddq1 == F.'*(u-c-g+M*(E*dA)*F*dq1);
% reduced_model = simplify(reduced_model,'Steps',500);
% sol_u = solve(reduced_model, u);
% 
% %%
% clc
% dq1=0; dq2=0; ddq1=0; ddq2=0;
% %IK("RR",[1,1],[0;1])
% q1=0.5236; q2=2.0944; g0 = 9.81;
% tau_eq = eval(sol_u)
% tau_eq = vpa(tau_eq,2)

%% Exam 11.01.2022 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_22.01.11.pdf]
% ex1
% clc
% syms q1 q2 q3 dq1 dq2 dq3 l1 l2 l3 m1 m2 m3 m_p Ip real
% dc1 = l1/2; dc2 = l2/2; dc3 = l3/2;
% 
% pc1 = [dc1*cos(q1); dc1*sin(q1)]; vc1 = diff_wrt(pc1,q1,1);
% pc2 = [l1*cos(q1)-(q2-dc2)*sin(q1); l1*sin(q1)+(q2-dc2)*cos(q1)]; vc2 = diff_wrt(pc2,[q1,q2],1);
% pc3 = [l1*cos(q1)-q2*sin(q1)+dc3*cos(q1+q3); l1*sin(q1)+q2*cos(q1)+dc3*sin(q1+q3)]; vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% p = [l1*cos(q1)-q2*sin(q1)+l3*cos(q1+q3); l1*sin(q1)+q2*cos(q1)+l3*sin(q1+q3)]; v = diff_wrt(p,[q1,q2,q3],1);
% 
% I1= 1/12*m1*l1^2; I2= 1/12*m2*l2^2; I3= 1/12*m3*l3^2;
% 
% T1= simplify(0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2);
% T2= simplify(0.5*m2*(vc2.'*vc2) + 0.5*I2*dq1^2);
% T3= simplify(0.5*m3*(vc3.'*vc3) + 0.5*I3*(dq1+dq3)^2);
% 
% Tp = simplify(0.5*m_p*(v.'*v) + 0.5*Ip*(dq1+dq3)^2);
% 
% [T,M] = kinetic_energy({T1,T2,T3,Tp},[dq1,dq2,dq3])
% 
% %%
% eqT1 = sym('T_1') == T1;
% eqT2 = sym('T_2') == T2;
% eqT3 = sym('T_3') == T3;
% eqTp = sym('T_p') == Tp;
% 
% raw2latex([eqT1,eqT2,eqT3,eqTp])
% %%
% [~,elems] = rewrite(M)
% %% M is the right one
% l1=0.45; l2=0.7; l3=0.35; m1=10; m2=10; m3=4; m_p=2; Ip=0.01;
% q1=pi/4; q2=0.25; q3=-pi/4;
% eval(J_pinv)
% %%
% syms q1 q2 q3 real
% J = jacobian(p,[q1,q2,q3]);
% J_pinv = pinv_w(J,M);
% %%
% clear
% syms q1 q2 q3 q4 r px py l1 l2 real
% clc
% p_obs = [px;py];
% a = [l1*cos(q1)-(q2-l2)*sin(q1); l1*sin(q1)+(q2-l2)*cos(q1)];
% b = p_obs + r*((a-p_obs)/norm(a-p_obs));
% q_vector = [q1,q2,q3];
% Ja = jacobian(a,q_vector);
% dq0 = H_func("obs",a,Ja,b);
% %%
% q1=pi/4; q2=0.25; q3=-pi/4; r=0.05; px=0.736; py=-0.1;
% dq1 = -2.1371; dq2=0.0818; dq3=-0.217;
% l1 = 0.45; l2=0.7;
% eval(dq0)

% ex3
% clear; clc
% syms q1 q2 dq1 dq2 d1 d2 L m1 m2 I2 g0 x_a x_b y_a y_b u real
% 
% pc1 = [0;q1-d1]; vc1 = [0;dq1];
% T1 = 0.5*m1*(vc1.'*vc1);
% 
% pc2 = [d2*cos(q2); q1+d2*sin(q2)]; vc2 = diff_wrt(pc2,[q1,q2],1);
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*dq2^2;
% 
% p = [L*cos(q2); q1+(L*sin(q2))];
% 
% [T,M] = kinetic_energy({T1,T2},[dq1,dq2]);
% 
% c = coriolis_terms(M);
% 
% g = gravity_terms([m1,m2],[0;-g0;0],{[pc1;0],[pc2;0]});
% 
% h = (p(2)-y_b)/(y_a-y_b) - (p(1)-x_b)/(x_a-x_b);
% 
% [dynamics,lambda,A,D,E,F] = reduced_dynamics(M,c,g,h)
% 
% u_sol = solve(dynamics,u)
% 
% %%
% syms t T real
% x_a = 0.7; x_b = 0.5; y_a = 2; y_b = 1;
% q2_a = acos(x_a);
% q1_a = y_a-sin(q2_a);
% q2_b = acos(x_b);
% q1_b = y_b-sin(q2_b);
% eqs = trajectory("cubicTimeCustom",[q1_a,q1_b,0,0,q2_a,q2_b,0,0],T,2,t);
% 
% %%
% syms u real
% I2=1.2; L=1; d2 = 0.6; q2 = q2_a; dq2=0; ddq2= eqs{2,3}; g0=9.81; m1=15; m2=8; t=0;
% 
% eval(u_sol)

%% Exam 10.09.2021 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_21.09.10.pdf]
% ex2
% syms q1 q2 q3 dq1 dq2 dq3 L m real
% 
% d=L/2;
% pc1 = [d*cos(q1); d*sin(q1)]; vc1 = diff_wrt(pc1,q1,1);
% pc2 = [L*cos(q1)+d*cos(q1+q2); L*sin(q1)+d*sin(q1+q2)]; vc2 = diff_wrt(pc2,[q1,q2],1);
% pc3 = [L*(cos(q1)+cos(q1+q2)) + d*cos(q1+q2+q3); L*(sin(q1)+sin(q1+q2))+d*sin(q1+q2+q3)]; vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% p = [L*(cos(q1)+cos(q1+q2)+cos(q1+q2+q3)); L*(sin(q1)+sin(q1+q2)+sin(q1+q2+q3))];
% 
% I=(1/12)*m*L^2;
% 
% T1 = 0.5*m*(vc1.'*vc1) + 0.5*I*dq1^2;
% T2 = 0.5*m*(vc2.'*vc2) + 0.5*I*(dq1+dq2)^2;
% T3 = 0.5*m*(vc3.'*vc3) + 0.5*I*(dq1+dq2+dq3)^2;
% 
% [T,M] = kinetic_energy({T1,T2,T3},[dq1,dq2,dq3])
% %%
% J = simplify(jacobian(p,[q1,q2,q3]));
% %%
% L = 0.5; m=5; q1=pi/2; q2=pi/2; q3=0;
% M_num = eval(M);
% J_pinv_num = eval(J_pinv);
% J_num = eval(J);
% 
% M_p = (J_num*M_num^-1*J_num.')^-1

%% Exam 04.02.2021 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_21.02.04.pdf]
%ex1
% clc
% syms q1 q2 q3 q4 dq1 dq2 dq3 dq4 m1 m2 m3 m4 I1 I2 I3 I4 dc1 dc3 dc4 a1 g0 real
% pc1 = [dc1*cos(q1); dc1*sin(q1)]; vc1 = diff_wrt(pc1,q1,1);
% pc2 = [a1*cos(q1); a1*sin(q1)]; vc2 = diff_wrt(pc2,q1,1);
% pc3 = [a1*cos(q1)+(q3-dc3)*cos(q1+q2); a1*sin(q1)+(q3-dc3)*sin(q1+q2)]; vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% pc4 = [a1*cos(q1)+q3*cos(q1+q2)+dc4*cos(q1+q2+q4); a1*sin(q1)+q3*sin(q1+q2)+dc4*sin(q1+q2+q4)]; vc4 = diff_wrt(pc4,[q1,q2,q3,q4],1);
% 
% T1 = 0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2;
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*(dq1+dq2)^2;
% T3 = 0.5*m3*(vc3.'*vc3) + 0.5*I3*(dq1+dq2)^2;
% T4 = 0.5*m4*(vc4.'*vc4) + 0.5*I4*(dq1+dq2+dq4)^2;
% 
% [T,M]= kinetic_energy({T1,T2,T3,T4},[dq1,dq2,dq3,dq4])
% 
% %%
% syms A_m1 A_m2 A_m3 A_m4 A_m5 A_m6 ddq1 ddq2 ddq3 ddq4 real
% c1 = I1 + I2 + I3 + I4 + a1^2*m2 + a1^2*m3 + a1^2*m4 + dc1^2*m1 + dc3^2*m3 + dc4^2*m4;
% c2 = I2 + I3 + I4 + dc3^2*m3 + dc4^2*m4;
% c3 = m3*dc3;
% c4 = m3 + m4;
% c5 = m4*dc4;
% c6 = I4 + m4*dc4^2;
% 
% rM = rewrite(M);
% M_a = subs(rM,c1,A_m1);
% M_a = subs(M_a,c2,A_m2);
% M_a = subs(M_a,c3,A_m3);
% M_a = subs(collect(M_a,q3^2),c4,A_m4);
% M_a = subs(collect(M_a,a1),c4,A_m4);
% M_a = subs(collect(M_a,q3),c4,A_m4);
% M_a = subs(M_a,c6,A_m6);
% M_a = subs(M_a,c5,A_m5);
% M_a = rewrite(M_a);
% 
% Y_A = jacobian(M_a*[ddq1;ddq2;ddq3;ddq4],[A_m1,A_m2,A_m3,A_m4,A_m5,A_m6])
% 
% %%
% g = gravity_terms([m1,m2,m3,m4],[0;-g0;0],{[pc1;0],[pc2;0],[pc3;0],[pc4;0]})
% 
% syms a1g a2g a3g a4g a5g a6g real
% c1 = a1*(m2+m3+m4) + m1*dc1;
% c2 = dc3*m3;
% c4 = dc4*m4;
% c3 = m3+m4;
% 
% rg = rewrite(g);
% g_a = subs(rg,c1,a1g);
% g_a = subs(collect(g_a,g0),c1,a1g);
% g_a = subs(collect(g_a,a1),c1,a1g);

%% Exam 12.07.2021 [https://www.diag.uniroma1.it/deluca/rob2_en/WrittenExamsRob2/Robotics2_21.07.12.pdf]
% ex1
% clc
% syms q1 q2 q3 L real
% 
% p = [L*(cos(q1)+cos(q1+q2)+cos(q1+q2+q3)); L*(sin(q1)+sin(q1+q2)+sin(q1+q2+q3))];
% 
% J = jacobian(p,[q1,q2,q3]);
% J_pinv = pinv(J);
% dJ = diff_wrt(J,[q1,q2,q3],1);
% 
% %%
% U_max = [3;3;3]
% 
% q1 = 0; q2 = 0; q3= pi;
% dq1 = pi/2; dq2 = -pi; dq3 = pi/2;
% L = 1;
% J_num = eval(J);
% J_pinv_num = pinv(J_num);
% dJ_num = eval(dJ);
% dq = [dq1;dq2;dq3];
% 
% u = -J_pinv_num*dJ_num*dq
% 
% ddp = J_num*u + dJ_num*dq

% ex3
% clear; clc
% syms q1 q2 q3 dq1 dq2 dq3 m1 m2 m3 dc1 dc2 dc3 I1 I2 I3 real
% 
% pc1 = [dc1*cos(q1); dc1*sin(q1)]; vc1 = diff_wrt(pc1,q1,1);
% pc2 = [(q2-dc2)*cos(q1); (q2-dc2)*sin(q1)]; vc2 = diff_wrt(pc2,[q1,q2],1);
% pc3 = [q2*cos(q1)+dc3*cos(q1+q3); q2*sin(q1)+dc3*sin(q1+q3)]; vc3 = diff_wrt(pc3,[q1,q2,q3],1);
% 
% T1 = 0.5*m1*(vc1.'*vc1) + 0.5*I1*dq1^2;
% T2 = 0.5*m2*(vc2.'*vc2) + 0.5*I2*dq1^2;
% T3 = 0.5*m3*(vc3.'*vc3) + 0.5*I3*(dq1+dq3)^2;
% 
% [T,M] = kinetic_energy({T1,T2,T3},[dq1,dq2,dq3]);
% c = coriolis_terms(M);
% 
% %%
% syms a1 a2 a3 a4 a5 real
% c1 = I1 + I2 + I3 + dc1^2*m1 + dc2^2*m2 + dc3^2*m3;
% c2 = I3 + m3*dc3^2;
% c3 = dc3*m3;
% c4 = m2 + m3;
% c5 = m2*dc2;
% 
% rM = rewrite(M);
% M_a = subs(rM,c1,a1);
% M_a = subs(M_a,c2,a2);
% M_a = subs(M_a,c3,a3);
% M_a = subs(M_a,c4,a4);
% M_a = subs(collect(M_a,q2),c4,a4);
% M_a = subs(M_a,c5,-a5);
% 
% %%
% rc = rewrite(c);
% c_a = subs(rc,c1,a1);
% c_a = subs(c_a,c2,a2);
% c_a = subs(c_a,c3,a3);
% c_a = subs(c_a,c4,a4);
% c_a = subs(collect(c_a,q2),c4,a4);
% c_a = subs(c_a,c5,-a5);
% %%
% syms ddq1_r ddq2_r ddq3_r dq1_r dq2_r dq3_r real
% coeffs_K = [a2,a3];
% coeffs_U = [a1,a4,a5];
% ddq_r = [ddq1_r;ddq2_r;ddq3_r];
% dq_r = [dq1_r;dq2_r;dq3_r];
% [S,~,~] = factorize_M(M_a);
% tau = M_a*ddq_r + S*dq_r + c;
% 
% Y_K = simplify(jacobian(tau,coeffs_K));
% Y_U = simplify(jacobian(tau,coeffs_U));