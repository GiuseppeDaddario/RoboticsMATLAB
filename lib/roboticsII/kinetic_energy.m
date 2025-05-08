function [T_tot,M] = kinetic_energy(joint_types, v, w)
joint_types = char(joint_types);
n_links = size(v,2);
T = cell(1,n_links);
% Syms for the masses and inertia
m_vector = sym('m', [n_links,1], 'real');
m = num2cell(m_vector); % {m1, m2, ..., }
I_vector = sym('I', [n_links,1], 'real');
I = num2cell(I_vector); % {I1, I2, ..., }
% Syms for the derivatives
dq_vector = sym('dq', [n_links,1], 'real');

for i=1:n_links
    linear = 0.5 * m{i} * v{i}.'*v{i};
    if joint_types(i)=='P'
        T{i} = linear;
    else
        T{i} = linear + (0.5 * w{i}.' * I{i} * w{i});
    end
end
T_tot = simplify(sum([T{:}]), 'Steps', 20);
M = simplify(hessian(T_tot, dq_vector), 'Steps', 20);
end