%% function for factorizing the inertia matrix using Christoffel's symbols

%% input variables:
% - M: inertia matrix

%% output variables:
% - S: factorization of the inertia matrix

function S = factorize_M(M)
n_links = size(M,1);
q_vector = sym('q', [n_links,1],'real');
dq_vector = sym('dq', [n_links,1],'real');
q = num2cell(q_vector);
c = cell(1,n_links);
S = cell(n_links,1);
for n=1:n_links
    term_1 = jacobian(M(:,n),q_vector);
    term_3 = diff(M,[q{n}]);
    c{n} = 1/2 * (term_1+term_1.'-term_3);
    S{n} = simplify(dq_vector.'*c{n});
end
S = vertcat(S{:});
dM = diff_matrix(M,q_vector);
disp("the result of the product (dM - 2S) is a skew symmetric matrix:")
simplify(dM - 2*S)
end
