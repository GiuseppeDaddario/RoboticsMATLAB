%% function for computing the time derivative of a vector

%% input variables:
% - v: the vector or the matrix to differentiate
% - q_vector: vector of the variables with respect to which differentiate

%% output variables:
% - dv: the resulting differentiated vector

function dv = diff_wrt(v, q_vector, grade)
n_links = numel(q_vector);
if grade==1
    dq_vector = sym('dq', [n_links,1], 'real');
elseif grade == 2
    dq_vector = sym('ddq', [n_links,1], 'real');
elseif grade == 0
    dq_vector = ones(1,n_links);
end
dv = sym(zeros(size(v)));

for k = 1:n_links
    dv = dv + diff(v, q_vector(k)) * dq_vector(k);
end

dv = simplify(dv);
end