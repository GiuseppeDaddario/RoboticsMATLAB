%% function for computing the gravity terms of the robot's dynamic model
%% input variables:
% - m: cell array of masses (e.g. {m1,m2,...})
% - rci: cell array containing the center of mass coordinates of each link (e.g. {rc1,rc2,...})
% - gv: column vector containing the gravity. (e.g. [0;g0;0])
%% output variables:
% - g: gravity terms of the robot's dynamic model

function g = gravity_terms(m,gv,rci)
n_links = numel(m);
q_vector = sym('q', [n_links,1],'real');
U=cell(1,n_links);

for i = 1:n_links
    U{i} = simplify(-1*m{i}*gv.'*rci{i});
end

U_tot = sum(cat(2, U{:}));
g=jacobian(U_tot,q_vector).';
g = simplify(g,'Steps',100);
end