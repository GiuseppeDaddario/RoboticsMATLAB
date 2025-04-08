%% function for computing the robot's dynamic model using the moving frames algorithm
%% input variables:
% - table: DH table of the robot
% - joint_types: string containing the types of the robot joints. (e.g. "RR")
% - CoMi: column vector containing the center of mass coordinates of each link
% - gv: column vector containing the gravity. (e.g. [0;g0;0])
% - renaming: bool indicating if the resulting matrix M should be renamed using a_i's coefficients (a1,a2,...)
%% output variables:
% - T: cell array containing the kinetic energies of all the links
% - M: inertia matrix
% - c: coriolis terms
% - g: gravity terms

function [T,M,c,g] = moving_frames(table,joint_types,CoMi,gv,renaming)
% init variables
joint_types = char(joint_types);
n_links = length(joint_types);
T = cell(1, n_links); w = cell(1, n_links); v = cell(1, n_links); vc = cell(1, n_links);
dq_vector = sym('dq', [n_links,1],'real');
dq = num2cell(dq_vector);
sigma = cell(1,n_links);
for i = 1:n_links
    if joint_types(i) == 'R'
        sigma{i} = 0;
    elseif joint_types(i) == 'P'
        sigma{i} = 1;
    end
end
rc = cell(1,n_links);
for i = 1:n_links
    idx = 3*(i-1) + 1;
    rc{i} = CoMi(idx:idx+2);
end
m = cell(1,n_links);
for i = 1:n_links
    var_name = sprintf('m%d', i);  % Building the string name [m1,m2,...]
    m{i} = sym(var_name);
end
Ic = cell(1,n_links);
for i = 1:n_links
    xx_name = sym(sprintf('Ic%d_xx', i),'real');  % Building the string name [Ici_xx]
    yy_name = sym(sprintf('Ic%d_yy', i),'real');  % Building the string name [Ici_yy]
    zz_name = sym(sprintf('Ic%d_zz', i),'real');  % Building the string name [Ici_zz]
    Ic{i} = [xx_name 0 0; 0 yy_name 0; 0 0 zz_name];
end

% getting the rotation matrices using DH table
[~,~,~, A] = DK(table); clc
R = cell(1,n_links);
for i = 1:n_links
    R{i} = A{i}(1:3, 1:3);
end

% computing values in frame i
for i = 1:n_links
    if (i==1)
        w{i} = simplify(R{i}.'*((1-sigma{i})*[0;0;dq{i}]),'Steps', 10);
        v{i} = simplify(R{i}.'*((sigma{i}*[0;0;dq{i}]+cross(w{i},A{i}(1:3,4)))),'Steps', 10);
    else
        w{i} = simplify(R{i}.'*(w{i-1}+((1-sigma{i})*[0;0;dq{i}])),'Steps', 10);
        v{i} = simplify(R{i}.'*(v{i-1}+((sigma{i}*[0;0;dq{i}])+cross(w{i},A{i}(1:3,4)))),'Steps', 10);
    end
    vc{i} = simplify(v{i} + cross(w{i},rc{i}),'Steps', 10);
    T{i} = simplify((1/2*m{i}*vc{i}'*vc{i}) + (1/2*w{i}'*Ic{i}*w{i}), 'Steps', 10);
end

% computing the total kinetic energy and extracting the inertia matrix M
T_tot = sum(cat(2, T{:}));
M = simplify(hessian(T_tot,dq_vector));

% renaming coefficients
if renaming
    M = rename_coefficients(M);
end

% computing Coriolis terms
c = coriolis_terms(M);

% computing gravity terms
g = gravity_terms(A,m,CoMi,gv);