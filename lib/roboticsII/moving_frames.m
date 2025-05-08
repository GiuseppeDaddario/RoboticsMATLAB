function [T, M, c,g] = moving_frames(table, joint_types, CoMi, I,gv)
joint_types = char(joint_types);
n_links = length(joint_types);

% Syms for the derivatives
dq_vector = sym('dq', [n_links,1], 'real');
dq = num2cell(dq_vector);  % dq{1}, dq{2}, ...

% Sigma (0 = R, 1 = P)
sigma = num2cell(double(joint_types == 'P'));

% Syms for the masses
m = cell(1, n_links);
for i = 1:n_links
    m{i} = sym(sprintf('m%d', i), 'real');
end

if isempty(I)
    % Syms for the inertia
    Ic = cell(1, n_links);
    for i = 1:n_links
        Ixx = sym(sprintf('Ic%d_xx', i), 'real');
        Iyy = sym(sprintf('Ic%d_yy', i), 'real');
        Izz = sym(sprintf('Ic%d_zz', i), 'real');
        Ic{i} = diag([Ixx, Iyy, Izz]);
    end
else
    Ic = I;
end

% Building center of masses from CoMi
rc = cell(1, n_links);
for i = 1:n_links
    rc{i} = CoMi(3*(i-1)+1 : 3*i);
end

% Direct Kinematics
[~, ~, ~, A] = DK(table); 
R = cell(1, n_links);
p = cell(1, n_links);
r_i = cell(1, n_links);
for i = 1:n_links
    R{i} = A{i}(1:3,1:3);
    p{i} = A{i}(1:3,4);
    r_i{i} = R{i}.' * p{i};
end

% Init for the first iteration
w = cell(1, n_links); v = cell(1, n_links); vc = cell(1, n_links); T = cell(1, n_links);
w_prev = sym([0;0;0]); v_prev = sym([0;0;0]);

for i = 1:n_links

    % Angular velocity
    w{i} = simplify(R{i}.' * (w_prev + (1 - sigma{i}) * [0; 0; dq{i}]), 'Steps', 10);

    % Velocity
    v_tmp = R{i}.' * (v_prev + sigma{i} * [0; 0; dq{i}]);

    v{i} = simplify(v_tmp + cross(w{i}, r_i{i}), 'Steps', 10);

    % Velocity of the center of mass
    vc{i} = simplify(v{i} + cross(w{i}, rc{i}), 'Steps', 10);

    % Kinetic energy
    T{i} = simplify(0.5 * m{i} * (vc{i}.' * vc{i}) + 0.5 * w{i}.' * Ic{i} * w{i}, 'Steps', 10);

    % Update previous values
    w_prev = w{i};
    v_prev = v{i};

    %displaying the computations
    disp_step(w,v,vc,T,i)

end

% Total kinetic energy
T_tot = simplify(sum([T{:}]), 'Steps', 20);

% Computing inertia matrix M
M = simplify(hessian(T_tot, dq_vector), 'Steps', 20);
fprintf("\nMatrix M(q):---------------------------------------------------------------------------------------------------------------------\n"); disp(M)

% TODO
% if renaming
%     M = rename_coefficients(M);
% end

% Coriolis and gravity terms
c = coriolis_terms(M);
fprintf("\nCoriolis terms c(q,dq):---------------------------------------------------------------------------------------------------------------------\n"); disp(c)

P = cell(1, n_links);
rci = cell(1,n_links);
P{1} = A{1};
for i = 2:n_links
    P{i} = P{i-1} * A{i};
end
for i = 1:n_links
    rci{i} = P{i}*[rc{i};1];
    rci{i}=rci{i}(1:3);
end
g = gravity_terms(m, gv,rci);
fprintf("\nGravity terms g(q):---------------------------------------------------------------------------------------------------------------------\n"); disp(g)
end
