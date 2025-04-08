%% function for computing the gravity terms of the robot's dynamic model
%% input variables:
% - A: cell array of homogeneus transformation matrices computed via direct kinematics
% - m: cell array of masses (e.g. {m1,m2,...}
% - CoMi: column vector containing the center of mass coordinates of each link
% - gv: column vector containing the gravity. (e.g. [0;g0;0])
%% output variables:
% - g: gravity terms of the robot's dynamic model

function g = gravity_terms(A,m,CoMi,gv)
n_links = numel(m);
q_vector = sym('q', [n_links,1],'real');
U=cell(1,n_links);
r0 = cell(1,n_links);
P = cell(1, n_links);
rc = cell(1,n_links);
for i = 1:n_links
    idx = 3*(i-1) + 1;
    rc{i} = CoMi(idx:idx+2);
end

P{1} = A{1};
for i = 2:n_links
    P{i} = P{i-1} * A{i};
end
for i = 1:n_links
    r0{i} = P{i}*[rc{i};1];
    U{i} = simplify(-1*m{i}*gv.'*r0{i}(1:3));
end
U_tot = sum(cat(2, U{:}));
g=jacobian(U_tot,q_vector).';
g = simplify(g,'Steps',100);
end