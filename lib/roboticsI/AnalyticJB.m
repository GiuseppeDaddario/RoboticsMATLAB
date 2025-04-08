function J = JB(rtype, R1, mode, vars, vals) 
syms q q1 q2 q3 l1 l2 l3 d1
if rtype == "RR"
    q = [q1, q2];
    r = [l1*cos(q(1)) + l2*cos(q(1)+q(2));
         l1*sin(q(1)) + l2*sin(q(1)+q(2));
         q(1) + q(2)];
    J = simplify(jacobian(r,q));
    return
elseif rtype == "RRP"
    q = [q1,q2,q3];
    r = [q(3) * cos(q(2)) * cos(q(1));
         q(3) * cos(q(2)) * sin(q(1));
         d1 + q(3)*sin(q(2))];
    J = simplify(jacobian(r,q));
    return
elseif rtype == "RRR"
    q = [q1,q2,q3];
    r = [cos(q(1))*((l2*cos(q(2))) + (l3*cos(q(2)+q(3))));
         sin(q(1))*((l2*cos(q(2))) + (l3*cos(q(2)+q(3))));
         d1 + (l2*sin(q(2))) + (l3*sin(q(2)+q(3)))];
    J = simplify(jacobian(r,q));
    return
else
    r = rtype;
    q = vars;
end

J = jacobian(r,q);


