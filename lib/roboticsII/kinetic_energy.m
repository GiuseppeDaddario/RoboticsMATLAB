function [T_tot,M] = kinetic_energy(T,dq_vector)
T_tot = simplify(sum([T{:}]), 'Steps', 20);
M = simplify(hessian(T_tot, dq_vector), 'Steps', 20);
end