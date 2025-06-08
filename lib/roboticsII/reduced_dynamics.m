function [dyn,lambda,A,D,E,F] = reduced_dynamics(M,c,g,h,D)
syms u real
n_links = size(M,1);
q = sym('q', [n_links,1], 'real');
dq = sym('dq', [n_links,1], 'real');
A = simplify(jacobian(h,q)); dA=diff_wrt(A,q,1);
if nargin == 4
    D = [0, A(1)^-1];
    M = [A; D];
    d = det(M);
    D = D / d;
end
dD = diff_wrt(D,q,1);
AD_T = M^-1;
E = AD_T(:,1);
F = AD_T(:,2);
v = D*dq; dv=diff_wrt(v,dq,2);

dyn = simplify((F.'*M*F)*dv == F.'*(u-c-g+M*(E*dA+F*dD)*F*v));
lambda = simplify(E.'*(M*F*dv-M*(E*dA+F*dD)*F*v+c+g-u));
end