function [J, A_saved, T_saved] = GeometricJB(table, joint_types)
alpha = sym('alpha');
d = sym('d');
a = sym('a');
theta = sym('theta');
N=size(table,1);
DHTABLE=table;

% Denavit-Hartenberg transformation matrix
TDH = [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
    sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
    0,            sin(alpha),            cos(alpha),            d;
    0,            0,                     0,                     1];


A = cell(1, N);
T = eye(4);
A_saved = cell(1, N);
T_saved = cell(1, N);

%Computing the joint matrices
for i = 1:N
    alpha = DHTABLE(i, 1);
    a = DHTABLE(i, 2);
    d = DHTABLE(i, 3);
    theta = DHTABLE(i, 4);
    A{i} = simplify(subs(TDH));
    T = simplify(T * A{i});
    fprintf('A{%d} :\n', i);
    disp(A{i})
    A_saved{i} = A{i};
    T_saved{i} = T;
    fprintf('T{%d} :\n', i);
    disp(T)
end

% Computing the jacobian
J = sym(zeros(6, N));
O_n = T_saved{N}(1:3, 4);

for i = 1:N
    if i == 1
        O_i = [0; 0; 0];        % Origine iniziale
        Z_i = [0; 0; 1];       % Asse Z iniziale
    else
        O_i = T_saved{i-1}(1:3, 4); % Origine del sistema precedente
        Z_i = T_saved{i-1}(1:3, 3); % Asse Z del sistema precedente
    end
    if joint_types(i) == 'R'
        J(1:3, i) = cross(Z_i, O_n - O_i);         %linear part
        J(4:6, i) = Z_i;                      %angular part
    elseif joint_types(i) == 'P'
        J(1:3, i) = Z_i;                      %linear part
        J(4:6, i) = [0; 0; 0];              %angular part
    end
end

J = simplify(J);
end