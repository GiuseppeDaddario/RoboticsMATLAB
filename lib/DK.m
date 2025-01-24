%Takes the Denavit-Hartenberg table and compute the direct kinematics.
%Display the point p and returns the final homogeneous matrix T0N (Denavit-Hartenberg matrix).
function [p,T0N] = DK(table)
alpha = sym('alpha');
d = sym('d');
a = sym('a');
theta = sym('theta');
N=size(table,1);
DHTABLE=table;

% Build the general Denavit-Hartenberg trasformation matrix
TDH = [ cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
        sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
          0             sin(alpha)             cos(alpha)            d;
          0               0                      0                   1];
fprintf("This is the general structure of the Denavit-Hartenberg matrix:\n\n");
disp(TDH);

fprintf("Now I build the transformation matrices of each joint substituting, for each row, the values of the Denavit-Hartenberg table in the matrix\n\n");
A = cell(1,N);
for i = 1:N
    alpha = DHTABLE(i,1);
    a = DHTABLE(i,2);
    d = DHTABLE(i,3);
    theta = DHTABLE(i,4);
    A{i} = subs(TDH);
    fprintf('A{%d} :\n', i)
    disp(A{i})
end

disp(['Number of joints N=',num2str(N)])
fprintf("\nNow I do the product of the matrices A{0}*A{1}*...A{n}\n\n")
T = eye(4);
for i=1:N 
    T = T*A{i};
    T = simplify(T);
    if i > 1
        indices = sprintf('%d', 1:i);
        fprintf('A{%s} :\n', indices);
        disp(T);
    end
end
fprintf("\nThe transformation of the end effector is given by the final matrix:\n")
% output TN matrix
T0N = T

fprintf("The coordinates of the end effector point are:\n")
% output ON position
p = T(1:3,4)

% output xN axis
n=T(1:3,1);
% output yN axis
s=T(1:3,2);
% output zN axis
a=T(1:3,3);