format short

%% Exam 19.09.2024
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
clc
