
format short;

% R = [
%     -1 0 0
%     0 -(1/sqrt(2)) -(1/sqrt(2))
%     0 -(1/sqrt(2)) 1/sqrt(2)
%     ]
% [theta, r] = getMatrixParameters(R)
%
%
%  R1 = generateMatrix(theta, [r(1,1) r(1,2) r(1,3)])

% a=2/3;
% b=-1/3;
%
% R = [b a a
%      a b a
%      a a b]
%
% [theta, r] = getMatrixParameters(R)

% R = [
%     0 1 0
%     1 0 0
%     0 0 -1
%     ]
% [theta, r] = getMatrixParameters(R)
%
%
%  R1 = generateMatrix(theta, [r(1,1) r(1,2) r(1,3)])
% R1 = generateMatrix(theta, r)

% clc
% order = ["X","Y","Z"];
% angles = [sym("alpha"),sym("phi")];
% A1 = getRfromEulerAngles(order)

%

% clc
% table = [ pi    -sym('L1')   0   sym('q1');
%             -pi/2  sym('L2')   0   sym('q2');
%          -pi/2  sym('L3')   0   sym('q3')];
% DK(table)

%real exercise
% clc
% R1= generateMatrix([1,0,0],sym('y'))
% R2=generateMatrix([1/sqrt(2),-1/sqrt(2),0], sym('d'))
% R3= R2*R1;
% R3 = simplify(R3)
% R3_numeric = subs(R3,[sym('y'),sym('d')],[-pi/2,pi/3]);
% R3_numeric = simplify(R3_numeric);
% R3_numeric = double(R3_numeric)
% [r, ang] = getMatrixParameters(R3_numeric);
% r = double(r)
% ang = double(ang)
%
% %checking the results
% R_check = generateMatrix(r,ang)

%real exercise
% clc
% q1 =sym('q1');
% q2 =sym('q2');
% table = [ pi   -0.5  0  q1;
%          -pi/2  0.6  0  q2];
%
% T0N = DK(table);
% T0N = subs(T0N,[q1,q2],[pi/2,-pi/2]);
% T0N = double(T0N)

%real exercise not completed
% clc
% syms i w v R L kv kt I F t tor di dt
% eq1 = diff(i,t)== -R/L*i - kv/L*w +1/L*v;
% eq2 = diff(w,t)== -F/I*w+kt/I*i;
% eq3 = tor == kt*i;
% [sol_i, sol_w, sol_tor] = solve([eq1, eq2, eq3], [i,w,tor]);
% disp('Soluzione per w:');
% disp(sol_w);
% disp('Soluzione per i:');
% disp(sol_i);
% disp('Soluzione per tor:');
% disp(sol_tor);

%real exercise
% clc
% syms a1 a3 q1 q2 q3
% table = [
%     -pi/2    a1   0   q1;
%     pi/2    0   q2   0;
%     0    a3   0  q3
%     ];
% T0N = DK(table)

%% 2nd midterm
% exercise
% clc
% Rd = 1/3 * [-2 2 -1; 2 1 -2; -1 -2 -2];
% [r, theta] = getMatrixParameters(Rd)
% %checking the results
% disp(Rd)
% R_res= generateMatrix(r(1,:),-theta)

% exercise
% clc
% order = ["Y","Z","Y"];
% R = getRfromEulerAngles(order)
% Ri = [0 0.5 -(sqrt(3)/2); -1 0 0; 0 sqrt(3)/2 0.5];
% Rf = [1 0 0; 0 0 1; 0 -1 0];
% Rt = transpose(Ri)*Rf %why the transpose? Because Rf = Ri * Rt so -> Ri(T) * Rf = [Ri(T) * Ri] * Rt !!!!!
% %we have two signs for a2, so for the rest
% alpha21 = atan2(-0.5000,-0.8660);
% alpha31 = atan2(0,0.5000/sin(alpha21));
% alpha11 = atan2(-0.5000/sin(alpha21),0);
% alpha22 = atan2(0.5000,-0.8660);
% alpha32 = atan2(0,0.5000/sin(alpha22));
% alpha12 = atan2(-0.5000/sin(alpha22),0);
% angles1 = [alpha11,alpha21,alpha31]
% angles2 = [alpha12,alpha22,alpha32]
% %checking the results
% R_res = getRfromEulerAngles(order,angles2);
% disp(Ri*R_res)
% disp(Rf)

% exercise Harmonic Drive
% clc
% L=0.7;
% Nfs = 160;
% nhd = Nfs/2;
% nw= 2;
% n = nhd * nw
% %the link is rotating CCK
% dr=0.1;
% dtheta=dr/(L*1000);
% %the resolution of the encoder should be less than perceived angle dtheta*n
% syms res nt nbit jl a tor
% eq1 = res == 2*pi/2^nt;
% [sol_nt] = solve(eq1,nt)
% dtn = dtheta*n;
% sol_nt = subs(sol_nt,res,dtn);
% disp(double(sol_nt)) %we have 9 tracks
% nturns = pi/(2*pi)*n %total range (pi*n) divided by 2pi (one full rotation)
% eq2 = 80 == 2^nbit;
% sol_nbit = solve(eq2,nbit);
% fprintf("nbits: %f\n",double(sol_nbit)) %7 bit
% jm = 1.2e-4;
% eq3 = (jm-jl/n^2)*a == 0;
% sol_jl = solve(eq3,jl);
% fprintf("Jl: %f\n", double(sol_jl))
% eq3 = tor == jm*n*7 + 1/n*(double(sol_jl)*7); %n*7 is the speed in the original config. so multiplied for the reduction ratio
% sol_tor=solve(eq3,tor);
% fprintf("torque: %f\n", double(sol_tor))

% exercise
% clc
% syms theta alpha a d d1 Te0
% eq = Te0 == [ cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
%         sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
%           0             sin(alpha)             cos(alpha)            d;
%           0               0                      0                   1]
% T0w=[1 0 0 -1; 0 -1 0 1; 0 0 -1 3.5; 0 0 0 1]
% Tcw=[1/sqrt(2) 0 -1/sqrt(2) 2; 0 -1 0 0; -1/sqrt(2) 0 -1/sqrt(2) 2; 0 0 0 1]
% Tec = [ 1 0  0 0;
%         0  -1 0 0;
%         0  0  -1 1;
%         0  0  0  1]
% Te0 = T0w^-1 * Tcw * Tec;
% Te0 = double(Te0)
% %The task is reduntant because is 5-dimensional and the robot is 6R -> we
% %have a parameter left free (alpha casually set to 0)

% exercise
% clc
% syms q1 q2 q3 H L
% table= [pi/2 0 H q1; pi/2 0 q2 pi/2; 0 L 0 q3];
% T0N = DK(table)

%% 3rd midterm
% exercise
% clc
% theta = pi/3;
% phi = -theta;
% r1= (1/sqrt(3))*[1 1 1];
% r2 = [0 1 0];
%
% R1 = generateMatrix(r1,theta);
% R2 = generateMatrix(r2,phi);
% R_tot = R2*R1

% exercise
% clc
% Ri = [0 0.5 -sqrt(3)/2; -1 0 0; 0 sqrt(3)/2 0.5];
% Rf = eye(3);
% order = ["Z","Y","Z"];
% R_tot = getRfromEulerAngles(order)
% Rt=Ri^-1*Rf;
% alpha21 = atan2(0.8660, 0.500);
% alpha22 = atan2(-0.8660, 0.500);
% alpha31 = atan2(0,0.8660/alpha21);
% alpha32 = atan2(0,0.8660/alpha22);
% alpha11 = atan2(0.8660/alpha21,0);
% alpha12 = atan2(0.8660/alpha22,0);
% angles1 = [alpha11 alpha21 alpha31]
% angles2 = [alpha12 alpha22 alpha32]
% %checking the results
% RC1 = getRfromEulerAngles(order, angles1);
% RC2 = getRfromEulerAngles(order,angles2);
% disp(Rf)
% disp(Ri*RC1)
% disp(Ri*RC2)

% exercise
% clc
% syms q1 q2 q3 q4 a4 d1 d3
% table = [-pi/2 0 d1 q1; pi/2 0 0 q2; -pi/2 0 d3 q3; 0 a4 0 q4];
% T0N = DK(table)

% exercise
% clc
% syms q1 q2 q3 q4
% table = [0 -1 0 q1;
%         0 1 0 q2;
%         0 1 0 q3;
%         0 1 0 q4];
% T0N = DK(table)

% experiment
% clc
% DH_table = [pi/2 0 1 pi/2; pi/2 0 2 pi/2; 0 1 0 pi/2];
% IK(["R","P","R"],DH_table)
% clc
% DHT = [pi/2 0.5 1 -pi/2 0; pi/2 1 0 pi/2 0; 0 1 0 0 pi/2];
% T = DK(DHT)
% q = IK("RRR",[1,1,1],T);
% disp(double(q))
% IK_auto(["R","R","R"],DH)

%% 4th midterm
% exercise 1
% clc
% order = ["Z","Y","X"];
% angles = [-pi/2,-pi/4,pi/4];
% %fixed axis so I flip the rotations
% R = getRfromEulerAngles(flip(order), flip(angles))
% % The unit vector r that is not rotated (nor scaled) by R is the eigenvector of R associated to its real eigenvalue λ= +1
% [r,theta] = getMatrixParameters(R);
% r=transpose(r);
% disp(r)
% r1 = R*r

% exercise 2
% clc
% Ri = [0 1 0; 0.5 0 sqrt(3)/2; sqrt(3)/2 0 -0.5];
% Rf = [1 0 0; 0 -1 0; 0 0 -1];
% R_ = Ri^-1 * Rf
% %I compute r in the initial reference frame
% [r, theta] = getMatrixParameters(R_)
% %checking the results
% %i have two solutions, taking + - [r,theta]
% R_check = generateMatrix(-r,-theta);
% disp(Ri*R_check)
% %the axis expressed in the final reference frame (after the rotation)
% r_fin = Rf * transpose(r)

% exercise 4
% clc
% syms m n
% eq1 = 27*(4-1)+9*m - m*(9*4) == 0;
% eq2 = 18*(4-1)+6*m - m*(6*4) == 0;
% [m_sol] = solve([eq1,eq2], [m])

% exercise 6
% clc
% syms N d4 q1 q2 q3 q4
% table = [0 0 q1 0; -pi/2 N 0 q2; 0 0 q3 0; 0 0 0 q4];
% T = DK(table)

% exercise 9
% clc
% L1 = 0.45;
% L2 = 0.35;
% r1 = 0.5/10;
% r2 = 0.25;
% Ne = 700;
% em = 4;
% lp_gen = 300;
% T = 1.2;
% res = 2*pi/Ne; %not considering amplification em
% De = res*lp_gen;
% n = r2/r1;
%
% D2 = De/n
% w = D2/T
%
% m_res=res/em;
% minD2 = m_res/n*L2 %dividing by the reduction ratio

% exercise 10
% clc
% R0w = generateMatrix([0,0,1],pi/2)
% p0w = [1;1;0]
% T0w = [R0w p0w; 0 0 0 1]
%
% Te0 = [0 0.5 -sqrt(3)/2 1; 1 0 0 -0.75; 0 -sqrt(3)/2 -0.5 1.5; 0 0 0 1]
%
% pte= [0;0.3;0.3]
% Rte = generateMatrix([1,0,0],-pi/2)
% Tte = [Rte pte; 0 0 0 1]
%
% Ttw = T0w*Te0*Tte

%% 5th midterm
% exercise 1
% clc
% Ri = [sqrt(2)/2 0 sqrt(2)/2; 0 -1 0; sqrt(2)/2 0 -sqrt(2)/2];
% order = ["Z","Y","X"];
% angles = [pi/3,pi/3,-pi/2];
% Rf = getRfromEulerAngles(flip(order),flip(angles))
% %r and theta expressed in frame i that leads to R(final)
% [r,theta] = getMatrixParameters(Rf)
% rb = Ri * transpose(r) %r expressed in base reference frame
% %check
% %if i express r in the base ref. frame i can compute the product of
% %matrices using the reverse order because both with fixed axes (the base
% %ones)
% sol1 = generateMatrix(rb, theta) * Ri
% sol2 = Ri * Rf

% exercise 2
% clc
% table = [-pi/2 0 128 0; 0 -612.7 0 pi/2; pi 571.6 0 pi; pi/2 0 163.9 -pi/2; -pi/2 0 -115.7 0; 0 0 92.2 0];
% T0N = DK(table)

% exercise 3
% clc
% syms q1 q2
% table = [0 0.4 0 q1; pi/2 0.9 0 q2];
% T0N = DK(table)
% beta = atan(A/B);
% %New Z axis (3rd row) is the old -x axis (1st col)
% R2e = [0 sin(beta) cos(beta);
%        0 cos(beta) -sin(beta);
%       -1       0       0];
% p1 = IK("RR",[0.4,0.5],[0;-0.9])
% p2 = IK("RR",[0.4,0.5],[-0.4;0.7])
% p3 = IK("RR",[0.4,0.5],[0;0])

%1. yes on the boundary
%2. we can compute the derivative of the angle in time
%3. good because allows to hold much heavier payloads, bad because reduces
%a lot the speed of the robot
%4. If we are in a singularity we cannot recover since a=0
%5. better (less) resolution for the incremental
% clc
% res1 = 360/(900*4);
% res1link = res1/40
% res2 = 0.0055
%6.

%% 6th midterm
% exercise1
% clc
% order = ["Y","X","Y"];
% angles = [pi/2,-pi/4,pi/4];
% r = [1/sqrt(3),-1/sqrt(3),1/sqrt(3)];
% eta = (-30*pi)/180;
% o1 = getRfromEulerAngles(order,angles)
% R0 = generateMatrix(r,eta)
% o0 = R0^-1 * o1 %the initial orientation, so i take the inverse!!!
% R = getRfromEulerAngles(flip(["X","Y","Z"]))
% disp(o0)
% c2_1 = sqrt(o0(3,2)^2 + o0(3,3)^2);
% c2_2 = -c2_1;
% s2 = -o0(3,1);
% a2_1 = atan2(s2,c2_1);
% a2_2 = atan2(s2,c2_2);
%
% c3_1 = o0(3,3)/c2_1;
% c3_2 = -c3_1;
% s3_1 = o0(3,2)/c2_1;
% s3_2 = -s3_1;
% a3_1 = atan2(s3_1,c3_1);
% a3_2 = atan2(s3_2,c3_2);
%
% c1_1 = o0(1,1)/c2_1;
% c1_2 = -c1_1;
% s1_1 = o0(2,1)/c2_1;
% s1_2 = -s1_1;
% a1_1 = atan2(s1_1,c1_1);
% a1_2 = atan2(s1_2, c1_2);
%
% a_sol = [a1_1 a2_1 a3_1; a1_2 a2_2 a3_2]

% exercise 2
% clc
% syms alpha a d theta
% TDH = [ cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
%         sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
%           0             sin(alpha)             cos(alpha)            d;
%           0               0                      0                   1];
% M = [-0.7071  0.5   -0.5    -1;
%      -0.7071 -0.5    0.5    -1;
%       0       0.7071 0.7071 -0.7071;
%       0       0      0       1];
% R = M(1:3,1:3);
% det(R); %it's a rotation matrix
% %I do the inverse taking the values from the matrix

%exercise 3
% clc
% table=[-pi/2 0      0    0;
%         pi/2 0      0    0;
%        -pi/2 0.045  0.55 0;
%         pi/2 -0.045 0    0;
%        -pi/2 0      0.3  0;
%         pi/2 0      0    0;
%         0    0      0.06 0];
% T0N = double(DK(table));
% R0B = eye(3);
% pOB = [0.022;0.014;0.0346];
% T0B = [R0B pOB; 0 0 0 1]

% exercise 4 motor
% clc
% Ra= 0.309;
% Fm=5e-5; %unità di misura SI!!!
% kt=7.88e-3;
% n_teeth=256;
% res = n_teeth/2;
% w=pi;
% %i set to zero the two derivatives
% syms Va i
% wm = -res * w %The minus is because the Harmonic Drive inverts the rotation !
% eq1 = Va == Ra*i + kt*wm;
% eq2 = kt*i == Fm*wm;
% [Va, i] = solve([eq1,eq2],[Va,i]);
% i = double(i);
% Va = double(Va)
%
% res_enc = res * 1e-3;
% n_turns = 360/res_enc;
% nbits = log(n_turns)/log(2); %19 bit
% fprintf("n bits: %d\n",12);
% fprintf("n pulses/turn: %d\n", n_turns/4);% quadrature electronics
% tot_pulse_onesec = (-wm/(2*pi))*n_turns
% %the counter goes down because the motor rotates CW and the link CCW, wm is
% %negative

%% 7th midterm
% exercise 1
% clc
% syms r1 r2 r3 N L q1 q2 q3
% Reb = generateMatrix([0,0,1],r3)
% %by geometric inspection
% peb = [
%     -((r2+N)*cos(pi-r1)+L*cos(pi-r3));
%     (r2+N)*sin(pi-r1)+L*sin(pi-r3);
%     0]
% table = [pi/2 0 0 q1; -pi/2 0 q2 0; 0 L 0 q3];
% Te0 = DK(table)
% pe0 = Te0(1:3,4)
% peb
% eq1 = pe0 == peb;% q= f(r) by inspection of these

% exercise 2
% clc
% n=0.35/0.08
% link_res= 0.01e-3/0.55; %dividing by the link lenght
% motor_res = n*link_res;
% pulses_per_turn = ((2*pi)/motor_res)/4
% n_bits=log(pulses_per_turn)/log(2)
%
% Tm = 0.8;
% Jl= 0.3025;
% %i compute the torque at the link
% Tl = Tm*n;
% Jm = Jl/n^2 %so I maximize the angular acceleration of the link
% %Tl = Jl*al + N(Jm*am) = (Jl+N^2*Jm)a = 2*Jl*al
% a = Tl /(2*Jl);
% al = 0.55*a %multiplying for the link lenght

% exercise 4
% clc
% Ra0 = [3/4 sqrt(3/8) -1/4; -sqrt(3/8) 1/2 -sqrt(3/8); -1/4 sqrt(3/8) 3/4];
% Rb0 = [1/sqrt(2) 0 -1/sqrt(2); 0 1 0; 1/sqrt(2) 0 1/sqrt(2)];
% R0a = Ra0^-1;
% Rba = R0a*Rb0;
% [r, theta] = getMatrixParameters(Rba)
% sol_theta = -theta
% axis_vector = -r
% %checking
% R_ = generateMatrix(r(2,:),theta(2,:))

%% 8th midterm
% exercise 2
% clc
% pAB = [3;7;-1];
% order = ["Z","Y","X"];
% angles=[pi/4,-pi/2,0];
% pBP = [1;1;0];
% Rba = getRfromEulerAngles(order, angles);
% %i use the affine relationship
% pAP = pAB + Rba * pBP

% exercise 3
% clc
% syms N M L q1 q2
% table = [pi/2 N 0 q1; 0 L M q2];
% T0e = DK(table);
% table2 = subs(table,[M,N,L,q1,q2],[2,0.3,1,pi/2,-pi/4])
% T0P= DK(table2);
% p = double(T0P(1:3,4))

% exercise 4
% clc
% syms a
% A = [-0.5 -a 0; 0 0 -1; a -0.5 0];
% a_num = sqrt(1-1/4);
% A = subs(A,a,a_num);
% [r, theta] = getMatrixParameters(A);
% r = double(r)
% theta = double(theta)

%% sketch planar 3R arm with unitary links (n=3) end-effector position in the plane (m=2)
% p =
%
% cos(q1 + q2 + q3) + cos(q1 + q2) + cos(q1)
% sin(q1 + q2 + q3) + sin(q1 + q2) + sin(q1)
%
% Jacobian (a 2x3 matrix)
%
% Jac =
%
% [- sin(q1 + q2 + q3) - sin(q1 + q2) - sin(q1), - sin(q1 + q2 + q3) - sin(q1 + q2), -sin(q1 + q2 + q3)]
% [  cos(q1 + q2 + q3) + cos(q1 + q2) + cos(q1),   cos(q1 + q2 + q3) + cos(q1 + q2),  cos(q1 + q2 + q3)]
% syms d1 q1 q2 q3
% table = [pi/2 0 d1 q1;
%          -pi/2 0 0 q2;
%          0 0 q3 0];
%
% T=DK(table);
% p = T(1:3,4);
% jacobian(p,[q1,q2,q3])

% [r, J] = JB("RR");
% names = {'q1','q2','l1','l2'};
% l= [1,2];
% q = [pi, pi/2];
% num = subs(r,names,[q,l])

% syms q1 q2 q3 q4
%
% r = [q2*cos(q1) + q4*cos(q1+q3);
%      q2*sin(q1) + q4*sin(q1+q3);
%      q1 + q3];
% J = JB(r,"Sym",[q1,q2,q3,q4],0)
%
% detJ = simplify(det(J * transpose(J)))
% eq1 = detJ == 0;
% ans = solve(eq1,[q2,q3])

%
% syms d1 q1 q2 q3
% r = [q3*cos(q2)*cos(q1);
%      q3*cos(q2)*sin(q1);
%      d1 + q3*sin(q2)];
% p = [1,1,1];
% q = [q1,q2,q3];
% q_val = [0,pi/2,0];
% d1_val = 0.5;
% tol = 1e-5;
% alpha = 0.7;
% max_iter = 20;
%
% q = gradientMethod(r,q,d1,q_val,d1_val,p,max_iter,tol,alpha);
% q = newtonMethod(r,q,d1,q_val,d1_val,p,max_iter,tol);
%
% syms a4 q1 q2 q3 q4
% th = [pi/2 0 0 q1;
%       pi/2 0 0 q2;
%      -pi/2 0 q3 0;
%         0 a4 0 q4];
% [J, As, Ts] = GeometricJB(th,['R','R','P','R']);
% R1 = transpose(As{1}(1:3,1:3));
% R1eye = [R1 zeros(3); zeros(3) R1];
% J1 = simplify(R1eye * J)
%
% syms q1 q2 q3 q4
%
% r = [q2*cos(q1) + q4*cos(q1+q3);
%      q2*sin(q1) + q4*sin(q1+q3);
%      q1 + q3];
% J = AnalyticJB(r,"Sym",[q1,q2,q3,q4],0)

%% Question 6 implement
% clc
% syms a b k q1 q2 q3 q4 c4 s4;
% table = [0 0 0 q1;
%     0 a 0 0;
%     0 q3 0 q2;
%     0 b 0 q4];
% [r, dk, R]= DK(table);
% r(3) = q1+q2+q4;
% J = AnalyticJB(r,R{1}, "Sym",[q1,q2,q3,q4],[]);
% Js = simplify(subs(J,[q2,q3],[pi/2,0]))
% rankJs = rank(Js)
% nullJs = null(Js)
%
% rangeJs1 = Js(1:3,1:2)
% rangeJs = simplify(R{1} * rangeJs1)

%% Question 7
% clc
% syms q1 q2
% r = [cos(q1) + cos(q1+q2);
%      sin(q1) + sin(q1+q2)];
% F = 10;
% J = AnalyticJB(r,[],"sym",[q1,q2],[]);
% Ja = J;
% Jb = J;
% Jas = simplify(subs(Ja,[q1,q2],[3*pi/4,-pi/2]))
% Jbs = simplify(subs(Jb,[q1,q2],[pi/2,-pi/2]))
%
% Fa = F *[cos(q1+q2);sin(q1+q2)];
% Fa = subs(Fa, [q1,q2],[3*pi/4,-pi/2])
%
% TauA = Jas' * Fa
%
% Fb = [-1 0;0 -1] * -Fa;
% TauB = double(Jbs' * Fb)

%% Question 8
% syms x y l1 l2 q1 q2
% beta = -20 * pi/180;
% eq1 = (x+0.8)*sin(beta) - (y-1.1)*cos(beta) == 0;
% eq2 = x^2 + y^2 == (0.9^2);
% [x_sol,y_sol] = solve([eq1,eq2],[x,y]);
% x_sol = double(x_sol);
% y_sol = double(y_sol);
%
% P2 = [x_sol(1); y_sol(1)]
% P1 = [x_sol(2); y_sol(2)]
%
% v = 0.3;
% T = 2;
%
% Prv = P1 + v*T*[cos(beta); sin(beta)]
% sol = IK('RR',[0.5,0.4],Prv)
%
% vg = v*[cos(beta);sin(beta)]
% J = AnalyticJB("RR");
% J = J(1:2,1:2);
% J = double(subs(J,[l1,l2,q1,q2],[0.5,0.4,1.5495,-1.0996]));
% qg = J^-1 * vg
%
% eqs = trajectory("cubic",[pi,1.5495,0,-0.4696,0,-1.0996,0,0.1987],2,2);
% eqs{1,1}
% eqs{1,2}
% eqs{2,1}
% eqs{2,2}
%
% pltEq(eqs,2)

%% Ex 1 samu
% clc
% Ri = [0 0 1; 0 1 0; -1 0 0];
% Rf = [sqrt(2)/2 0 sqrt(2)/2;
%       sqrt(2)/2 0 -sqrt(2)/2;
%       0 1 0];
%
% [R, Rs] = getRfromEulerAngles(["Z","X","Z"])
%
% s2_1 = sqrt(Ri(1,3)^2 + Ri(2,3)^2);
% s2_2 = -s2_1;
% c2 = Ri(3,3);
% a2_1 = atan2(s2_1,c2);
% a2_2 = atan2(s2_2,c2);
%
% s1_1 = Ri(1,3)/s2_1;
% s1_2 = -s1_1;
% c1_1 = -Ri(2,3)/s2_1;
% c1_2 = -c1_1;
% a1_1 = atan2(s1_1,c1_1);
% a1_2 = atan2(s1_2, c1_2);
%
% s3_1 = Ri(3,1)/s2_1;
% s3_2 = -s3_1;
% c3_1 = Ri(3,2)/s2_1;
% c3_2 = -c3_1;
% a3_1 = atan2(s3_1,c3_1);
% a3_2 = atan2(s3_2,c3_2);
%
% anglesRi = [a1_1 a2_1 a3_1; a1_2 a2_2 a3_2];
%
% s2_1 = sqrt(Rf(1,3)^2 + Rf(2,3)^2);
% s2_2 = -s2_1;
% c2 = Rf(3,3);
% a2_1 = atan2(s2_1,c2);
% a2_2 = atan2(s2_2,c2);
%
% s1_1 = Rf(1,3)/s2_1;
% s1_2 = -s1_1;
% c1_1 = -Rf(2,3)/s2_1;
% c1_2 = -c1_1;
% a1_1 = atan2(s1_1,c1_1);
% a1_2 = atan2(s1_2, c1_2);
%
% s3_1 = Rf(3,1)/s2_1;
% s3_2 = -s3_1;
% c3_1 = Rf(3,2)/s2_1;
% c3_2 = -c3_1;
% a3_1 = atan2(s3_1,c3_1);
% a3_2 = atan2(s3_2,c3_2);
%
% anglesRf = [a1_1 a2_1 a3_1; a1_2 a2_2 a3_2];
%
%
% Ti = [0 cos(anglesRi(1,1)) sin(anglesRi(1,1))*sin(anglesRi(1,2));
%      0 sin(anglesRi(1,1)) -cos(anglesRi(1,1))*sin(anglesRi(1,2));
%      1 0 cos(anglesRi(1,2))]
%
% Tf = [0 cos(anglesRf(1,1)) sin(anglesRf(1,1))*sin(anglesRf(1,2));
%      0 sin(anglesRf(1,1)) -cos(anglesRf(1,1))*sin(anglesRf(1,2));
%      1 0 cos(anglesRf(1,2))]
%
% phi_dot_i= Ti^-1 * [0;1;0]
% phi_dot_f= Tf^-1 * [0;0;1]
%
% sol = trajectory('cubic',[1.5708,0.7854,0,1,1.5708,1.5708,1,0,-1.5708,0,0,0],1.5,3)
% pltEq(sol,1.5)
%
%  eqs = sol(:);
%  T = 1.5;
%  s = 0.75;
% % Definisci la variabile simbolica s e t
%     syms t
%
%     % Sostituisci t con s/T nelle equazioni
%     phi = subs(eqs{1}, t, s/T);
%     theta = subs(eqs{2}, t, s/T);
%     psi = subs(eqs{3}, t, s/T);
%
%     phi_dot = subs(eqs{4}, t, s/T);
%     theta_dot = subs(eqs{5}, t, s/T);
%     psi_dot = subs(eqs{6}, t, s/T);
%
%     % Matrice di trasformazione T(phi, theta)
%     T_matrix = [0, cos(phi), sin(phi)*sin(theta);
%                 0, sin(phi), -cos(phi)*sin(theta);
%                 1, 0, cos(theta)];
%
%     % Calcolo della velocità angolare
%     omega = T_matrix * [phi_dot; theta_dot; psi_dot];
%
%     % Converti il risultato in una forma più leggibile
%     omega = vpa(omega,5);
%
%     % Mostra il risultato
%     disp('Velocità angolare omega(t):');
%     disp(omega);

%% Question 9
% clc
% syms t T
% q1 = pi/4 + pi/4 * (3*(t/T)^2 - 2*(t/T)^3);
% q2 = -pi/2 * (1-cos((pi*t)/T));
%
% q1_vel = diff(q1,t);
% q2_vel = diff(q2,t);
%
% q1_acc = diff(q1_vel,t);
% q2_acc = diff(q2_vel,t);
%
% boundaries = [subs(q1,t,0) subs(q1,t,T);
%             subs(q1_vel,t,0) subs(q1_vel,t,T);
%             subs(q1_acc,t,0) subs(q1_acc,t,T);
%             subs(q2,t,0) subs(q2,t,T);
%             subs(q2_vel,t,0) subs(q2_vel,t,T);
%             subs(q2_acc,t,0) subs(q2_acc,t,T)]
%
% V1 = 4;
% V2 = 8;
% A1 = 20;
% A2 = 40;
%
% tv1 = abs(subs(q1_vel, t, T/2));
% tv2 = abs(subs(q2_vel,t,T/2));
% ta1 = abs(subs(q1_acc,t,T));
% ta2 = abs(subs(q2_acc,t,T));
%
% eq1 = tv1 == V1;
% eq2 = tv2 == V2;
% eq3 = ta1 == A1;
% eq4 = ta2 == A2;
%
% times = double([solve(eq1,T);
%     solve(eq2,T);
%     solve(eq3,T);
%     solve(eq4,T)])
%
% pltPVA({q1,q2,q1_vel,q2_vel,q1_acc,q2_acc},0.6226);

%% Question 10
% clc
% syms l1 l2 q1 q2 pd
% pd_dot = 0.3 * [cos(-20*pi/180);sin(-20*pi/180)]
% fk = [l1*cos(q1) + l2*cos(q1+q2);
%       l1*sin(q1) + l2*sin(q1+q2)];
%
% e_p = pd - fk;
% R = generateMatrix([0,0,1],-20*pi/180);
% R = R(1:2,1:2);
% e_task = R' * e_p;
%
% J = AnalyticJB(fk,[],"sym",[q1,q2],[]);
% K=[3 0; 0 10;];
%
% q_dot = J^-1 * (pd_dot + R*K*e_task);
% vpa(simplify(q_dot),4)

%% Question 5
% clc
% syms q1 q2 e alpha
% assume(e,'real');
% r = [q2 * cos(q1);
%      q2 * sin(q1)];
% q0 = [pi/4; e];
% p1 = [-1;1];
% p2= [1;1];
%
%
% %p1
% x_curr = subs(r,[q1,q2],q0');
% err1 = simplify(p1 - x_curr);
% err2 = simplify(p2 - x_curr);
% J = jacobian(r, [q1,q2]);
% J_current = subs(J,[q1,q2],q0');
%
% %newton method p1
% delta_theta_1 = pinv(J_current) * err1;
% delta_theta_2 = pinv(J_current) * err2;
% q1_n = q0 + delta_theta_1;
% q1_n_2 = q0 + delta_theta_2;
% q1_n = simplify(q1_n)
% q1_n_2 = simplify(q1_n_2)
%
% %gradient method p1
% gradient = J_current' * err1;
% gradient_2 = J_current' * err2;
% q1_g = q0 + alpha * gradient;
% q1_g_2 = q0 + alpha * gradient_2;
% q1_g = simplify(q1_g)
% q1_g_2 = simplify(q1_g_2)

%% Question 6
% clc
% syms q1 q2 q3 h qdot
% d = 20*pi/180;
% table = [d 0 0 q1;
%     -pi/2  0 h q2;
%          0 0 0 q3];
% % J = GeometricJB(table,["R","R","R"]);
% Ja = simplify(J(4:6,:))
% detJ = simplify(det(Ja));
% eq1 = detJ == 0;
% sol = solve(eq1,[q1,q2],"ReturnConditions",true)
%
% [n, dk, Rs] = DK(table)
%
% eq2 = Ja*qdot == 0;
% solve(eq2,qdot)

%% Question 7
% clc
% Ri = [0.5 0 -sqrt(3)/2;
%      -sqrt(3)/2 0 -0.5;
%      0 1 0];
%
% Rf = [sqrt(2)/2 -sqrt(2)/2 0;
%      -0.5 -0.5 -sqrt(2)/2;
%       0.5 0.5 -sqrt(2)/2];
%
% getRfromEulerAngles(["X","Y","Z"])
%
% s2 = Ri(1,3);
% c2 = sqrt(Ri(1,1)^2 + Ri(1,2)^2);
% a2 = atan2(s2,c2);
% c2_2 = -c2;
% a2_2 = atan2(s2,c2_2);
%
% s1 = Ri(2,3)/-c2;
% c1 = Ri(3,3)/c2;
% a1 = atan2(s1,c1);
% s1_2 = -s1;
% c1_2 = -c1;
% a1_2 = atan2(s1_2,c1_2);
%
% s3 = Ri(1,2)/-c2;
% c3 = Ri(1,1)/c2;
% a3 = atan2(s3,c3);
% s3_2 = -s3;
% c3_2 = -c3;
% a3_2 = atan2(s3_2,c3_2);
%
% anglesRi = [a1 a2 a3; a1_2 a2_2 a3_2]
%
% s2 = Rf(1,3);
% c2 = sqrt(Rf(1,1)^2 + Rf(1,2)^2);
% a2 = atan2(s2,c2);
% c2_2 = -c2;
% a2_2 = atan2(s2,c2_2);
%
% s1 = Rf(2,3)/-c2;
% c1 = Rf(3,3)/c2;
% a1 = atan2(s1,c1);
% s1_2 = -s1;
% c1_2 = -c1;
% a1_2 = atan2(s1_2,c1_2);
%
% s3 = Rf(1,2)/-c2;
% c3 = Rf(1,1)/c2;
% a3 = atan2(s3,c3);
% s3_2 = -s3;
% c3_2 = -c3;
% a3_2 = atan2(s3_2,c3_2);
%
% anglesRf = [a1 a2 a3; a1_2 a2_2 a3_2]
%
% syms alpha beta
% T = [1 0 sin(beta);
%      0 cos(alpha) -cos(beta) * sin(alpha);
%      0 sin(alpha) cos(alpha) * cos(beta)];
%
% Vi = 0;
% Vf = [3; -2; 1];
% Ai = 0;
% Af = 0;
%
% phi_dot_f = (subs(T,[alpha,beta],[anglesRf(1,1),anglesRf(1,2)]))^-1 * Vf;
%
% sol=trajectory("quintic", [1.5708,2.3562,0,3,0,0,-1.0472,0,0,2.12,0,0,0,0.7854,0,0.707,0,0],1,3);
% pltPVA(sol,1)

%% Question 8
% clc
% q_in = [-pi/9 11*pi/18 -pi/4];
% q3_f = -pi/2;
%
% syms q1 q2 q3
%
% table = [0 1 0 q1;
%          0 1 0 q2;
%          0 1 0 q3];
% r = DK(table);
% p = double(subs(r,[q1,q2,q3],q_in));

%% Exercise 2
% clc
% syms a b c w t V1 V2 A1 A2
% pd = [c + a*sin(2*w*t);
%       c + b*sin(w*t)];
%
% pd_dot = simplify(diff(pd, t))
% pd_ddot = simplify(diff(pd_dot,t))

%% Exercise 3
% clc
% syms a1 a2 a3 da1 da2 da3
% % R = getRfromEulerAngles(["Y","X","Y"])
% Rdot = diff(R,a1)*a1 + diff(R,a2)*da2 + diff(R,a3)*da3;
% assume(da1,"real")
% assume(da2,"real")
% assume(da3,"real")
% assume(a1,"real")
% assume(a2,"real")
% assume(a3,"real")
% S = simplify(Rdot * R');
% w = [S(3,2);S(1,3);S(2,1)];
% w = simplify(w)
%
% T = [0 cos(a1) sin(a1)*sin(a2);
%      1 0 cos(a2);
%      0 -sin(a1) cos(a1)*sin(a2)];

%% Exercise 4
% clc
% syms q1 q2 q3 t
% v = 0.5;
% v_task = v * [ cos(pi/2);sin(pi/2);0];
% B = [0.75 1.8];
% A = [3 2.5];
% T = norm(B-A)/v;
% pd = vpa(A + ((v*t)/norm(B-A))*(B-A),3)
% pd_t = double(subs(pd,t,1));
% phi_d = atan2((A(2)-B(2)),(A(1)-B(1))) + pi/2;
% rd_dot = double(diff([pd phi_d],t));
% pd_dot = double([diff(pd,t),0]);
%
% table = [0 2 0 q1;
%          0 2 0 q2;
%          0 2 0 q3];
%
% fk = DK(table);
% fk(3) = q1+q2+q3;
% J = simplify(jacobian(fk,[q1,q2,q3]))
% qt_dot = J^-1 * pd_dot';
%
% pt2 = pd_t - 2*([cos(phi_d), sin(phi_d)]);
% q12 = IK("RR",[2,2],pt2);
% q12 = q12(2,:)';
% q3 = phi_d - (q12(1)+q12(2));
% q = [q12;q3]
%
% qt_dot_t = double(subs(qt_dot,[t,q1,q2,q3],[1,q']))

%% Exercise 1
% clc
% syms t
%
% R = [cos(t) 0 sin(t);
%      sin(t)^2 cos(t) -sin(t)*cos(t);
%      -sin(t)*cos(t) sin(t) cos(t)^2];
%
% [w,w_dot,S] = skew(R)

%% Exercise 3
% clc
% syms q1 q2 q3
% tab1 = [0 1 0 q1;
%         0 1 0 q2];
%
% w_T_A = [eye(3) [-2.5 1 0]';
%         [0 0 0] 1];
% Rb = generateMatrix([0,0,1],pi/6);
% w_T_B = [Rb [1 2 0]';
%         [0 0 0] 1];
% [n,A_T_E,l] = DK(tab1);
% A_T_E =  double(subs(A_T_E,[q1,q2],[pi/3,-pi/2]));
% E_T_E = [-1 0 0 0;
%          0 -1 0 0;
%           0 0 1 0;
%           0 0 0 1];
% B_T_E = w_T_B^-1* w_T_A * A_T_E * E_T_E;
%
% p = B_T_E(1:2,4);
%
% phi_Bd = atan2(B_T_E(2,1),B_T_E(1,1))
% p_t2=p-[cos(phi_Bd); sin(phi_Bd)]
% px=p_t2(1);
% py=p_t2(2);
% c2=(px^2+py^2-2)/2
% s2=sqrt(1-c2^2) % sign + on sqrt results in elbow up solution (arbitrary choice)
% q_B2=atan2(s2,c2)
% s1=py*(1+c2)-px*s2 % denominator (> 0) discarded in s1 and c1
% c1=px*(1+c2)+py*s2
% q_B1=atan2(s1,c1)
% q_B3=phi_Bd-(q_B1+q_B2)

%% Exercise 4
% clc
% syms q1 q2 q3 l1 l2 l3
% assume([q1,q2,q3,l1,l2,l3],"real")
% J = [-sin(q1)*(l2*cos(q2) + l3*cos(q3)) -l2*cos(q1)*sin(q2) -l3*cos(q1)*sin(q3);
%      cos(q1)*(l2*cos(q2) + l3*cos(q3)) -l2*sin(q1)*sin(q2) -l3*sin(q1)*sin(q3);
%      0 l2*cos(q2) l3*cos(q3)];
%
% R = generateMatrix([0,0,1],q1);
% J_1 = simplify(R' * J);
%
% Js = simplify(subs(J,[q2,q3],[pi/2,pi/2]))
% analysisJB(Js)
% analysisJB(Js')

%% Exercise 5
% clc
% M = 2;
% U_max = 8;
% xi = 0;
% xi_dot = -2;
% xf = 3;
% xf_dot = 0;
%
% A_max = U_max/M;

%% Exercise 2
% clc
% syms q1 q2 q3
% table = [0 0 0 q1;
%          0 q2 0 0;
%          0 L 0 q3];
%
% fk = DK(table);
% fk(3) = q1 + q3;
% p = [2; 0.4; -pi/2];
% alpha = -pi/2;
% L = 0.6;
% J = jacobian(fk,[q1,q2,q3])
%
% q2_r = sqrt((2-L*cos(alpha))^2+(0.4-L*sin(alpha))^2);
% q1_r = atan2(0.4-L*sin(alpha),2-L*cos(alpha));
% q3_r = alpha - q1_r;
% qi = [q1_r,q2_r,q3_r]
%
% v_task = -2.5 * [0; -sin(alpha);0];
% J = double(subs(J,[q1,q2,q3],[q1_r,q2_r,q3_r]));
% q_dot = double(J^-1 * v_task)
% tau = double(J' * [-15;0;-6])

%% Exercise 3
% clc
% A = [1,1,1];
% B = [-1,5,0];
% T = 2.5;
%
% Ra = [0 1 0;
%       1 0 0;
%       0 0 -1];
%
% Rb = [-1/sqrt(2) 0 1/sqrt(2);
%       0 -1 0;
%       1/sqrt(2) 0 1/sqrt(2)];
%
% R = Ra' * Rb;
% [r,theta] = getMatrixParameters(R);
%
% syms t
% s = 6*(t/T)^5 - 15*(t/T)^4 + 10*(t/T)^3;
% s_dot = diff(s,t);
% s_ddot = diff(s_dot,t);
%
% pltPVA({s,s_dot,s_ddot},T)
%
% R_mid = generateMatrix(r(1,:),theta(1)/2);
% R_mid = Ra * R_mid

%% Exercise 3
% clc
% syms q1 q2 q3 q1_dot q2_dot q3_dot px py
% table = [0 1 0 q1;
%          0 1 0 q2;
%          0 1 0 q3];
% fk = DK(table);
% fk(3) = q1+q2+q3;
% Ja = jacobian(fk(1:2,:));
% J = jacobian(fk);
% detJ = simplify(det(J))
% q_dot =[q1_dot;q2_dot;q3_dot];
% p_ddot = [px;py];
% q_ddot = simplify(J^-1 * ([p_ddot;0]-[diff(Ja)*q_dot;0]))
%
% q_num = double(subs(q_ddot,[q1,q2,q3,q1_dot,q2_dot,q3_dot,px,py],[pi/4,pi/3,-pi/2,-0.8,1,0.2,1,1]))

%% Exercise 4
% clc
% syms S lam L T
% S = 0;
% lam = 3;
% L = 1;
% T = 4;
% qs = [S;pi/2];
% qg = [S+lam;pi/2];
%
% p_mid = [S+(lam/2);L/4];
% q1_mid = (p_mid(1) - sqrt(L^2 - p_mid(2)^2));
% q2_mid = atan2(p_mid(2),p_mid(1)-q1_mid);
% qm = [q1_mid;q2_mid];
%
% [spline1,spline2] = build_spline(0,0,qs,qm,qg,T)

%% Exercise 3
% clc
% syms q1 q2
% l1=0.5;
% l2=0.4;
% fk = [l1*cos(q1)+l2*cos(q1+q2);
%       l1*sin(q1)+l2*sin(q1+q2)];
% p = [0.4 -0.3];
% tol = 1e-4;
%
% newtonMethod(fk,[q1,q2],[],[0.3491,-2.0944],[],p,3,tol)

%% Exercise 4
% clc
% syms a4 q1 q2 q3 q4
% assume([a4,q1,q2,q3,q4], "real")
% table=[pi/2 0 0 q1;
%        pi/2 0 0 q2;
%       -pi/2 0 q3 0;
%         0 a4 0 q4];
% [J,A_saved] = GeometricJB(table,["R","R","P","R"])
% R1 = A_saved{1};
% R1 = R1(1:3,1:3);
% J(1:3,:) = R1' * J(1:3,:);
% J(4:6,:) = R1' * J(4:6,:);
% J = simplify(J)

% analysisJB(J',4,[q1,q2,q3,q4])

%% Exercise 2
% clc
% syms q1 q2 q3
%
% table = [0 0 0 q1;
%       pi/2 0 q2 pi/2;
%          0 0 q3 0];
%
% fk = DK(table);
%
% J = jacobian(fk)

%% Exercise 3
% clc
% syms q1 q2 s tau t
% l1=2;
% l2=1;
% assume([q1,q2],"real")
% table = [0 l1 0 q1;
%          0 l2 0 q2];
%
% fk = DK(table);
% fk(3) = [];
%
% J = jacobian(fk,[q1,q2]);
%
% Ja = simplify(subs(J,[q1,q2],[pi/2,pi]));
% Jb = simplify(subs(J,[q1,q2],[0,0]));
%
% p_dot_a = [5;0];
% p_dot_b = [0;-1];
%
% q_dot_a = double(pinv(Ja) * p_dot_a)
% q_dot_b = double(pinv(Jb) * p_dot_b)
% %
% T = 2;
% tr_t = trajectory("cubic",[pi/2,0,-5/2,-3/10,pi,0,5/2,-1/10],T,2)
% % pltTau("space",t,1)
%
% tr_s= trajectory("cubicTime",[0,1,0,0],T,1)
% % pltTau("time",s,2)
%
% tr = pltTau("customTime",tr_t,T,tr_s{1});
% sdot = subs(tr_s{1},tau,t/T);
% tr{3} = tr{3} * diff(sdot,t);
% tr{4} = tr{4} * diff(sdot,t);
% new_tr = {tr{1},tr{2},tr{3},tr{4}};
% pltTau("customTime",new_tr,T,tr_s{1})
%
% %rescaling time
% smax = findMax(tr_s{2},tau,0,1)
% q1_max = findMax(tr_t{3},s,0,1) %change
% q2_max = findMax(tr_t{4},s,0,1)
%
% %take the max divided by the V constraint and then multiply by smax, so you got the new time

%% Exercise 4
% clc
% syms q1 q2 q3 s
% assume(s,"real")
% table=[pi/2 0 0.7 q1;
%        0 0.5 0 q2;
%        0 0.5 0 q3];
% T = 3;
% r = 0.5;
% h0 = 0.2;
% h = 0.4;
% v = 1;
%
% px = r*cos(2*pi*s);
% py = r*sin(2*pi*s);
% pz = h0+h*s;
% pd = [px;py;pz];
%
% fk = DK(table);
% j = simplify(jacobian(fk,[q1,q2,q3]))
%
% %s in 0,3
% t_in = double(subs(pd,s,0));
% t_f = double(subs(pd,s,3));
% distance_min = norm(t_in - [0;0;0.7]);
% distance_max = norm(t_f - [0;0;0.7]);
%
% q_in = [0, pi/6, -pi/2];
% p_in = double(subs(fk,[q1,q2,q3],q_in))
%
% error = t_in - p_in
%
% pd_dot = diff(pd,s)
% [T, N, B, kappa, tau] = frenetFrame(pd, s)
%  R = [T N B]
%  R = simplify(R);
% K = [2 0 0;
%      0 5 0;
%      0 0 5];
% J0 = double(subs(j,[q1,q2,q3],q_in))
% pd_real = double(subs(pd_dot,s,0))
%
% R_dot = diff(R,s);
% w0 = [0;0;6.2832];
% S_e = [0; -1.1499; 0];

%% Esercise 4
% clc
% syms q1 q2 e
% 
% fk = [q2*cos(q1);
%     q2*sin(q1)];
% q0 = [pi/4;e];
% pd1 = [-1;1];
% pd2 = [1;1];
% 
% x_current = subs(fk,[q1,q2],[q0(1),q0(2)]);
% disp("x_current = "), disp(x_current);
% 
% % Compute the error
% error = pd2 - x_current;
% 
% % Compute the Jacobian
% J = jacobian(fk, [q1,q2]);
% J_current = simplify(subs(J, [q1, q2], [q0(1),q0(2)]));
% 
% % Update theta
% delta_theta = pinv(J_current) * error;
% theta = q0 + delta_theta;
% simplify(theta)

%% Exercise 2
% clc
% syms q1 q2 q3 q4 la lb fz
% assume([q1,q2,q3,q4,la,lb,fz],"real")

% table = [0 0 q1 0;
%      -pi/2 0 0 q2;
%      pi/2 0 la q3;
%       0 lb 0 q4];
% 
% fk = DK(table);
% fk = simplify(fk)
% vars = [q1,q2,q3,q4,la,lb];
% vals1 = [0,0,0,0,0.5,0.75];
% vals2 = [1,0,-pi/2,pi/2,0.5,0.75];
% 
% q_1 = double(subs(fk,vars,vals1))
% q_2 = double(subs(fk,vars,vals2))
% 
% J = GeometricJB(table,["P","R","R","R"])

% JL = simplify(J(1:3,:))
% JA = simplify(J(4:6,:))
% 
% F = [0;0;fz];
% 
% tau = -JL' * F
% tau_1 = simplify(subs(tau,vars,vals1))
% tau_2 = simplify(subs(tau,vars,vals2))

% detJ = simplify(det(JA * JA'))

%% Exercise 3
% clc
% syms q1 q2 q3 L
% 
% fk = [q1 + L*cos(q3);
%       q2 + L*sin(q3);
%       q3];
% J = jacobian(fk,[q1,q2,q3])

%% Exercise 2
% clc
% syms q1 q2 q3 q4 d1 a2 a3 d4
% assume([q1,q2,q3,q4,d1,a2,a3,d4],"real")
% table= [-pi/2 0 d1 q1;
%         0 a2 0 q2;
%         -pi/2 a3 0 q3;
%         0 0 d4 q4];
% 
% [J,Ts,n] = GeometricJB(table,["R","R","R","R"])
% JL = J(1:3,:);
% JA = J(4:6,:);

% JL = arrayfun(@factor, JL, 'UniformOutput', false)
% JL = cellfun(@(c) prod(c(:)), JL)
% JA = arrayfun(@factor, JA, 'UniformOutput', false)
% JA = cellfun(@(c) prod(c(:)), JA)

% R1 = Ts{1}(1:3,1:3);
% JL = simplify(R1'*JL);
% JA = simplify(R1'*JA);
% 
% detJL = analysisJB(JL(:,1:3));
% detJA = analysisJB(JA)

%% Exercise 3
% clc
% syms v t R q1 q2
% assume([v,t,R],"real")
% R = [-sin(v*t/R) cos(v*t/R);
%      -cos(v*t/R) -sin(v*t/R)];
% Rdot = diff(R',t);
% S = simplify(Rdot*R)
% 
% q0 = IK("RR",[0.5,0.5],[0.35;0.3])
% p0_dot = [3;3];
% 
% l1= 0.5; l2=0.5;
% fk = [l1*cos(q1) + l2*cos(q1+q2);
%       l1*sin(q1) + l2*sin(q1+q2)];
% 
%  J = simplify(jacobian(fk,[q1,q2]))
%  J = double(subs(J,[q1,q2],[q0(1),q0(2)]))
% 
%  q0_dot = J^-1 * p0_dot

%% Exercise 4
% clc
% 
% syms psi q1 q2 q3
% assume([q1,q2,q3],"real")
% 
% Re3 = [0 sqrt(2)/2 sqrt(2)/2;
%        0 sqrt(2)/2 -sqrt(2)/2;
%        -1 0 0];
% 
% table = [ -pi/2 1 0 q1;
%         pi/2 0 q2 0;
%          0 sqrt(2) 0 q3];
% 
% [fk,t0n, Rs] = DK(table)
% fk(3) = q1 + q3;
% J = simplify(jacobian(fk,[q1,q2,q3]));
% J0 = double(subs(J,[q1,q2,q3],[pi/2,-1,0]))
% 
% R30 = double(subs(Rs{3},[q1,q3],[pi/2,0]));
% Re0 = R30 * Re3
% 
% Re0t = [Re0 zeros(3);
%        zeros(3) Re0];
% Ft = [0;-1;-2;2;0;0];
% 
% 
% a = (Re0t * Ft)

%% Exercise 5
% clc
% syms t a b v t
% a = 1;
% b = 0.3;
% p = [-a*sin(2*pi*t);
%      b*cos(2*pi*t)];
% 
% p_dot = diff(p,t);
% p_ddot = diff(p_dot,t);

% pltPVA({p(1) p_dot(1) p_ddot(1);p(2) p_dot(2) p_ddot(2)},1)
% 
% pt = subs(p,t,v*t);
% pt_dot = diff(pt,t);
% pt_ddot = diff(pt_dot,t);
% 
% pt = subs(pt,v,2);
% pt_dot = subs(pt_dot,v,2);
% pt_ddot = subs(pt_ddot,v,2);
% 
% pltPVA({pt(1) pt_dot(1) pt_ddot(1);p(2) pt_dot(2) pt_ddot(2)},1/2)

%% Exercise 3
% clc
% syms q1 q2 q3 q4 a2 a3 a4 d1
% assume([q1,q2,q3,a2,a3,a3,d1],"real")

% table = [pi/2 0 d1 q1;
%         0 a2 0 q2;
%         0 a3 0 q3;
%         0 a4 0 q4];
% 
% [fk,t0n,Rs] = DK(table);
% 
% J = GeometricJB(table,["R","R","R","R"]);
% 
% JL = J(1:3,:);
% JA = J(4:6,:);

% JL1 = simplify(Rs{1}' * JL)
% H = [1 0 0 0; 0 1 0 0; 0 -1 1 0; 0 0 -1 1];
% JLH = simplify(JL1 * H);
% % singularities when q3 and q4 {0, +- pi}
% detJLH = analysisJB(JLH);

% analysisJB(JA)

% JA1 = simplify(Rs{1}' * JA);
% J1 = [JLH;JA1];
% J1(3:4,:) = [];
% analysisJB(J1)

% JLf = subs(JL,[a2,a3,a4,q1,q2,q3,q4],[210.5,268,174.5,0,0,pi/2,0])
% double(null(JLf))
% q_dot = [0; 0; -0.0021*174.5; 0.0021*174.5 + 0.0021*268];
% JLf * q_dot

%% Exercise 4
% clc
% syms p1 p2 p3 p4 t1 t2 t3 t4 v_in v_fin
% p1 = 45; p2 = 90; p3 = -45; p4 = 45;
% v_in = 0; v_fin = 0;
% t1 = 1; t2=2; t3=2.5; t4=4;

% [sol,coeffs,spA,spB,spC,spA_dot,spB_dot,spC_dot,spA_ddot,spB_ddot,spC_ddot] = spline3(v_in,v_fin,p1,p2,p3,p4,t1,t2,t3,t4);
% pltSplines(spA,spB,spC,spA_dot,spB_dot,spC_dot,spA_ddot,spB_ddot,spC_ddot,t1,t2,t3,t4)

%scaling time
% Vmax = 308.1115;
% V = 250;
% k = Vmax/V;
% 
% t2n = t1 + k*(t2-t1);
% t3n = t1 + k*(t3-t1);
% t4n = t1 + k*(t4-t1);
% [sol,coeffs,spA,spB,spC,spA_dot,spB_dot,spC_dot,spA_ddot,spB_ddot,spC_ddot] = spline3(v_in,v_fin,p1,p2,p3,p4,t1,t2n,t3n,t4n);
% pltSplines(spA,spB,spC,spA_dot,spB_dot,spC_dot,spA_ddot,spB_ddot,spC_ddot,t1,t2n,t3n,t4n)

%% Exercise 2
% clc
% syms q1 q2
% 
% p = [q2*cos(q1);
%     q2 * sin(q1)];
% 
% Pi = [4;3];
% Pf = [-1;1];
% V = [2;2.5];
% A = [3;1.5];
% 
% qi = [atan2(Pi(2),Pi(1));
%       sqrt(Pi(1)^2 + Pi(2)^2)];
% qf = [atan2(Pf(2),Pf(1));
%       sqrt(Pf(1)^2 + Pf(2)^2)];
% 
% delta = double(qf - qi)
% 
% %checking if there is the cruising phase for q1
% abs(delta(1)) > V(1)^2/A(1)
% Ta1 = V(1)/A(1);
% T1 = (abs(delta(1))*A(1) + V(1)^2)/(A(1)*V(1));
% 
% %checking if there is the cruising phase for q2
% abs(delta(2)) > V(2)^2/A(2) %negative, negative vel. and acceleration
% Ta2 = sqrt(abs(delta(2))/A(2));
% T2 = 2*Ta2;
% 
% T_min = max(T1,T2)
% 
% %so we need to scale down q1
% k = T_min/T1;
% V(1) = V(1)/k;
% A(1) = A(1)/k^2;
% 
% J = jacobian(p,[q1,q2]);
% detJ = simplify(det(J));

%% Exercise 3
% clc
% syms l q1 q2 q3
% table = [0 l 0 q1;
%          0 l 0 q2;
%          0 l 0 q3];
% 
% p = DK(table)
% J = jacobian(p,[q1,q2,q3])
% 
% %pinv = J.T * (J J.T)^-1
% xt = ([-2;4]-[4;2])/(norm([-2;4]-[4;2]));
% yt = [-xt(2);xt(1)];
% R = [xt yt];
% 
% q0 = [-pi/2;0;pi/2];
% P0 = double(subs(p,[q1,q2,q3,l],[q0',2]))
% J0 = double(subs(J,[q1,q2,q3,l],[q0',2]));
% K = [1.1512 0; 0 3.4538];
% er = [2;6];
% q_res = double(pinv(J0(1:2,:))*(R*K*R'*er))

%% Exercise 3
% clc
% syms q1 q2 q3 q4 l1 l2 l3 l4
% assume([q1,q2,q3,q4,l],"real")
% 
% pm = [l2*cos(q1+q2)+l1*cos(q1);
%       l2*sin(q1+q2)+l1*sin(q1)];
% 
% pe = pm + [l4*cos(q1+q2+q3+q4)+l3*cos(q1+q2+q3);
%       l4*sin(q1+q2+q3+q4)+l3*sin(q1+q2+q3)];
% 
% 
% ve = [0.2;0];
% vm=[-0.2;0.1];
% q = [pi/3;pi/6;0;-pi/2];
% 
% JM = jacobian(pm,[q1,q2]);
% JEM = jacobian(pe,[q1,q2]);
% JEE = jacobian(pe,[q3,q4]);
% 
% J = [JM zeros(2);
%      JEM JEE];

%% Exercise 4
% clc
% 
% M = 0.4;
% L = 1.6;
% V = 1;
% A = 2;
% 
% Ta_1 = sqrt(M/A); %tempo accelerazione se non c'è coast
% Ta_2 = V/A; %tempo accelerazione se c'è coast
% T_1 = 2*Ta_1; %tempo totale se non c'è coast
% T_2 = (L*A + V^2)/(A*V); %tempo totale se c'è coast
% 
% 
% T_tot = 2*(T_1+T_2)

%% Exercise 1
% clc
% syms q1 q2 q3 l2 l3
% 
% table = [0 q1 0 0;
%          0 l2 0 q2;
%          0 l3 0 q3];
% 
% [fk,t0n,Rs] = DK(table);
% 
% r = [fk(1:2);q2+q3];
% 
% J = jacobian(r,[q1,q2,q3]);
% 
% Js = subs(J,q2,pi/2);
% analysisJB(Js)
% 
% J = jacobian(fk,[q1,q2,q3]);
% J = Rs{1}' * J;
% J = J(1:2,:);
% analysisJB(J)
% 
% Js = simplify(subs(J,[q2,q3],[pi/2,pi]));

%% Exercise 2
% clc
% syms q1 q2 q3
% 
% p = [q3*cos(q2);
%      q3*sin(q2);
%      q1];
% 
% pd = [1;-1;3];
% qa = [-2 0.7*pi sqrt(2)];
% qb = [2 pi/4 sqrt(2)];
% tol = 0.1;
% 
% sol_a = newtonMethod(p,[q1,q2,q3],[],qa,[],pd,3,tol)
% sol_b = newtonMethod(p,[q1,q2,q3],[],qb,[],pd,3,tol)

%% Exercise 3
% clc
% l1=1.2; l2=0.8;
% q0 = [0;0];
% v = 1.5;
% delta = 15*pi/180;
% p0 = [-2;1];
% prv = [0;1.5359];
% d = norm(prv-p0);
% T = d/v;

% table = [0 l1 0 q1;
%          0 l2 0 q2];
% fk = DK(table);
% r = fk(1:2);
% J = jacobian(r,[q1,q2]);
% J = double(subs(J,[q1,q2],[2.1122,-1.4250]));
% q_dot = J^-1 * [v*cos(delta);v*sin(delta)];

% t = trajectory("cubicTime",[0,2.1122,0,q_dot(1),0,-1.4250,0,q_dot(2)],T,2);
% pltTau("time",t,T)

%% Exercise 5
% clc
% Pi = [0.6;-0.4]; Pf = [1;1];
% Pi_prime = [-2;0]; Pf_prime = [2;2];
% 
% qi = IK("RR",[1,1],Pi);
% qf = IK("RR",[1,1],Pf);
% 
% fk = [cos(q1)+cos(q1+q2); sin(q1)+sin(q1+q2)];
% J = jacobian(fk,[q1,q2]);
% 
% qi_prime = double(subs(J,[q1,q2],[qi(1),qi(2)]))^-1 * Pi_prime
% qf_prime = double(subs(J,[q1,q2],[qf(1),qf(2)]))^-1 * Pf_prime

% tr = trajectory("cubic",[qi(1),qf(1),qi_prime(1),qf_prime(1),qi(2),qf(2),qi_prime(2),qf_prime(2)],1,2);
% ts = trajectory("cubicTime",[0,1,0,0],3,1);
% 
% new_tr = pltTau("customTime",tr,3,ts{1})

%% Exercise 4
% clc
% syms a1 a2 a3 d1 q1 q2 q3
% 
% table = [pi/2 a1 d1 q1;
%     0 a2 0 q2; 
%     pi/2 a3 0 q3];
% 
% [J,As,Ts] = GeometricJB(table,["R","R","R"]);
% JL = J(1:3,:);
% JL1 = simplify(As{1}(1:3,1:3)' * JL);
% 
% JA = J(4:6,:);
% double(subs(Ts{3},[q1,q2,q3,a1,a2,a3,d1],[0,pi/2,0,0.04,0.445,0.04,0.33])*[0;0;0.52;1]);

%% Exercise 2 Robotics1_23.06.12
% derivate quando il tempo è cubico rest to rest, (metà cerchio)

%% Exercise 3 Robotics1_23.06.12 Matrice H
% clc
% syms q1 q2 q3 q4 d1 a1 d2 a3 a4
% assume([q1 q2 q3 q4 d1 a1 d2 a3 a4],"real")
% table=[0 a1 d1 q1;
%        pi/2 0 d2 q2;
%        0 a3 0 q3;
%        0 a4 0 q4];
% 
% [J,As,Ts] = GeometricJB(table,["R","R","R","R"]);
% JL = J(1:3,:);
% JA = J(4:6,:);
% 
% JL2 = simplify(Ts{2}(1:3,1:3)'* JL);
% JA2 = simplify(Ts{2}(1:3,1:3)'* JA);
% J2 = [JL2;JA2];
% J2H = J2 * [1 0 0 0;
%            -1 1 0 0;
%             0 0 1 0;
%            0 0 -1 1];

%% Exercise 4
% clc
% syms q1 q2 q3 q4
% r = [q2*cos(q1)+q4*cos(q1+q3);
%     q2*sin(q1)+q4*sin(q1+q3);
%     q1+q3];
% 
% J = simplify(jacobian(r,[q1,q2,q3,q4]))

%% Exercise 5
% clc
% syms r s h s_dot s_ddot
% assume(r>0); assume(s>0); assume(h>0); assume(s_dot,"real"); assume(s_ddot,"real")
% R = [0 1 0; 0 0 1; 1 0 0];
% p = simplify([0;0;r] + (R * [r*cos(s);r*sin(s);h*s]));
% [t,n,b] = frenetFrame(p,s);
% 
% p_prime = diff(p,s);
% p_dot = p_prime*s_dot;
% p_ddot = diff(p_prime,s)*s_dot^2 + p_prime*s_ddot;

% V=2; A=4.5; h=0.3; r=0.4;
% min_s_dot = min(V/sqrt(h^2+r^2),sqrt(A/r));
% min_s_ddot = A/sqrt(h^2+r^2);

% [T,Ta] = bang_bang(4*pi,min_s_dot,min_s_ddot)

%% Exercise pinhole camera
clc
syms a1 a2 L H X1 X2 Z1 Z2
assume([a1,a2,L,H],"real")
Tc1_b = [cos(a1) 0 sin(a1) L;
         sin(a1) 0 -cos(a1) H;
         0 1 0 0;
         0 0 0 1];

Tc2_b = [cos(a2) 0 sin(a2) L;
         sin(a2) 0 -cos(a2) -H;
         0 1 0 0;
         0 0 0 1];

Tc2_c1 = simplify(Tc1_b^-1 * Tc2_b);

A = [0.01;0;-0.006;0.00006]'; B = [0.012;0;-0.002;-0.000024]';
P1 = [X1;0;Z1;1]; P2 = [X2;0;Z2;1];

P2_1 = Tc2_c1*P2;
P2_1 = subs(A*P1,[X1,Z1],[P2_1(1),P2_1(3)]);
[x2,z2] = solve([P2_1,B*P2],[X2,Z2]);
x2 = double(subs(x2,[a1,a2,H,L],[pi/3,2*pi/3,0.4,0.8]));
z2 = double(subs(z2,[a1,a2,H,L],[pi/3,2*pi/3,0.4,0.8]));

X2_num = [x2;0;z2;1]
Tc2_c1 = subs(Tc2_c1,[a1,a2],[pi/3,2*pi/3]);
X1_num = double(subs(Tc2_c1*X2_num,[H,L],[0.4,0.8]))




