% Friday 22 November 2024. Midterm of Robotics I
% Giuseppe D'Addario
format short;

% exercise 1
% clc
% order_i = ["Z","X","Y"];
% angles_i = [pi/2,pi/4,-pi/4];
% r_f=[0,-sqrt(2)/2,sqrt(2)/2];
% theta_f = pi/6;
% Ri = getRfromEulerAngles(order_i,angles_i)
% Rf = generateMatrix(r_f, theta_f)
% Rif = Ri^-1 * Rf
% order_f = ["Y","X","Y"];
% Rif_rpy = getRfromEulerAngles(flip(order_f))
% 
% s2_1 = sqrt(Rif(1,2)^2 + Rif(3,2)^2);
% s2_2 = -s2_1;
% c2 = Rif(2,2);
% a2_1 = atan2(s2_1,c2);
% a2_2 = atan2(s2_2,c2);
% 
% s1_1 = Rif(1,2)/s2_1;
% s1_2 = -s1_1;
% c1_1 = Rif(3,2)/s2_1;
% c1_2 = -c1_1;
% a1_1 = atan2(s1_1,c1_1);
% a1_2 = atan2(s1_2, c1_2);
% 
% s3_1 = Rif(2,1)/s2_1;
% s3_2 = -s3_1;
% c3_1 = -(Rif(2,3)/s2_1);
% c3_2 = -c3_1;
% a3_1 = atan2(s3_1,c3_1);
% a3_2 = atan2(s3_2,c3_2);
% 
% angles = [a1_1 a2_1 a3_1; a1_2 a2_2 a3_2]
% %checking
% R_1 = getRfromEulerAngles(flip(order_f),flip([a1_1,a2_1,a3_1]));
% R_2 = getRfromEulerAngles(flip(order_f),flip([a1_2,a2_2,a3_2]));
% disp(R_1)
% disp(R_2)

% exercise 2
% clc
% R = getRfromEulerAngles(["X"], [pi/2]);
% syms q1 q2 q3 L
% table = [pi/2 0 -q1 pi/2; -pi/2 0 q2 -pi/2; 0 L 0 q3];
% T0e = DK(table);
% pe0 = T0e(1:3,4)
% pew = [1 0 0;0 0 -1; 0 1 0]*pe0
% exercise 3
% clc
% rm = 0.5e-2;
% re = 40e-2;
% rl = 10e-2;
% n = (rm/re)*(re/rl);
% n^2/7e-4;


% exercise 4
clc
syms q1 q2 q3 q4 alpha px py pz L
eq1 = px == sin(q1)*q3 + L*cos(q1)*cos(q4);
eq2 = py == -cos(q1)*q3 + L*sin(q1)*cos(q4);
eq3 = pz == q2 + L*sin(q4);
eq4 = alpha == q4;

[sol_q1,sol_q2,sol_q3,sol_q4] = solve([eq1,eq2,eq3,eq4], [q1,q2,q3,q4])
[a1,a2,a3,a4] = subs([sol_q1,sol_q2,sol_q3,sol_q4],[px,py,pz,alpha],[2,2,4,-pi/4])

% exercise 5
% clc

% exercise 6
% clc

% exercise 7
% clc

% exercise 8
% clc

% exercise 9
% clc

% exercise 10
% clc