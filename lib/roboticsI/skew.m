function [w,w_dot,S] = skew(R)
syms t
assume(t,"real");
R_dot = diff(R,t);
S = simplify(R_dot*R');
w = [S(3,2);S(1,3);S(2,1)];
w_dot = diff(w,t);
end