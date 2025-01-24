function J = JB(rtype, mode, vars, vals) 
syms q q1 q2 q3 l1 l2 l3 d1
if rtype == "RR"
    q = [q1, q2];
    r = [l1*cos(q(1)) + l2*cos(q(1)+q(2));
         l1*sin(q(1)) + l2*sin(q(1)+q(2));
         q(1) + q(2)];
elseif rtype == "RRP"
    q = [q1,q2,q3];
    r = [q(3) * cos(q(2)) * cos(q(1));
         q(3) * cos(q(2)) * sin(q(1));
         d1 + q(3)*sin(q(2))];
elseif rtype == "RRR"
    q = [q1,q2,q3];
    r = [cos(q(1))*((l2*cos(q(2))) + (l3*cos(q(2)+q(3))));
         sin(q(1))*((l2*cos(q(2))) + (l3*cos(q(2)+q(3))));
         d1 + (l2*sin(q(2))) + (l3*sin(q(2)+q(3)))];
else
    r = rtype;
    q = vars;
end

J = jacobian(r,q);

% Compute numerical jacobian if data are given
if mode == "Num"
    J = subs(J,vars,vals)
end

% Show the determinant, the null space and the rank of J
if size(J,1) ~= size(J,2)
    detJ = simplify(det(J * transpose(J)))
else
    detJ= simplify(det(J))
end
nullJ = simplify(null(J))
rankJ = rank(J)

%Try to compute the singularities (almost never works due to trigonometric)
eq1 = detJ == 0;
sol = solve(eq1,vars);
results = struct2cell(sol);
disp('Singularities when:');
for i = 1:numel(vars)
    fprintf('%s: %s\n', char(vars(i)), char(results{i}));
end
