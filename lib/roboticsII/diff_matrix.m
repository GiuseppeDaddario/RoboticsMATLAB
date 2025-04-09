%% function for computing the time derivative of a matrix

%% input variables:
% - A: the matrix to differentiate
% - q_vector: vector of the variables with respect to which differentiate

%% output variables:
% - dA: the resulting differentiated matrix

function dA = diff_matrix(A,q_vector)
n_links = numel(q_vector);
dq_vector = sym('dq', [n_links,1],'real');
dA = sym(zeros(n_links));
for k = 1:n_links
    dA = dA + diff(A, q_vector(k)) * dq_vector(k);
end
dA = simplify(dA);
end