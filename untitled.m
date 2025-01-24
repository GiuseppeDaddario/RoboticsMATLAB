
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

syms q1 q2 q3 q4

r = [q2*cos(q1) + q4*cos(q1+q3);
     q2*sin(q1) + q4*sin(q1+q3);
     q1 + q3];
J = AnalyticJB(r,"Sym",[q1,q2,q3,q4],0)









