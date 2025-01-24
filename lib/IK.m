%Function for computing the inverse kinematics of 3 types of robot:
% "RR,RRP,RRR".
%rtype   -> is the string containing one of the above specified type
%llength -> are the link lenghts, for RRP put just 'd1'.
%point   -> is the end effector point. You can also put here the DH table
function q = IK(rtype, llength, point)
%check if point is the 3vector or the DH table
if size(point,1) == 4
    p = point(1:3,4);
else
    p = point;
end

if rtype == "RR"
    disp("Robot type: RR")
    fprintf("We compute q2 by squaring and summing the equations of the direct kinematics\n")
    l1=llength(1);
    l2=llength(2);
    px = p(1);
    py = p(2);
    sum = (px+py)^2-(l1+l2)^2;
    c2 = (px^2+py^2-(l1^2+l2^2))/(2*l1*l2);
    if c2 > 1 || c2 < -1
        disp("The point is out of the WS1");
        q = [];
        return
    end
    s2 = sqrt(1-c2^2);
    s2p = -s2;
    q2 = atan2(s2,c2);
    q2p = atan2(s2p,c2);
    if l1==l2 && c2 == -1
        disp("Singular case: there are infinity to the power of 1 solutions");
        q = [];
        return
    end
    fprintf("We use an algebraic solution for computing q1\n")
    q1 = atan2(py*(l1+l2*c2)-(px*l2*s2),px*(l1+l2*c2)+(py*l2*s2));
    q1p = atan2(py*(l1+l2*c2)-(px*l2*s2p),px*(l1+l2*c2)+(py*l2*s2p));
    if abs(sum) <1e-16
        disp("The point is on the boundary of the WS1: one solution");
        q = [q1 q2];
        return
    end
    disp("Since we are in a regular case, we have two solutions")
    q = [q1 q2; q1p q2p];
    return
elseif rtype == "RRP"
    disp("Robot type: RRP")
    px = p(1);
    py = p(2);
    pz = p(3);
    d1 = llength;
    disp("We only take the positive value of qr after sqaring the eq. given from the direct kinematics")
    q3 = sqrt(px^2+py^2+(pz-d1)^2);
    if abs(q3) < 1e-16
        disp("Singular case: there are infinity to the power of 2 solutions")
        q = [];
        return
    end
    c2 = sqrt(px^2+py^2)/q3;
    c2p = -c2;
    s2 = (pz-d1)/q3;
    q2 = atan2(s2,c2);
    q2p = atan2(s2,c2p);
    if abs(px^2+py^2) < 1e-16
        disp("Singular case: there are infinity to the power of 1 solutions")
        q = [];
        return
    end
    q1 = atan2(py/c2,px/c2);
    q1p = atan2(py/c2p, px/c2p);
    disp("Since we are in a regular case, we have two solutions")
    q = [q1 q2 q3; q1p q2p q3];
    return
elseif rtype == "RRR"
    disp("Robot type: RRR")
    disp("We compute q3 from the eq. of the direct kinematics and we take the two values of the sqare root")
    px = p(1);
    py = p(2);
    pz = p(3);
    d1 = llength(1);
    l2 = llength(2);
    l3 = llength(3);
    c3 = (px^2+py^2+((pz-d1)^2)-l2^2-l3^2)/(2*l2*l3);
    if c3 > 1 || c3 <-1
        disp("The point is out of the WS1")
        q = [];
        return
    end
    s3 = sqrt(1-c3^2);
    s3p = -s3;
    q3 = atan2(s3,c3);
    q3p = atan2(s3p,c3);
    if abs(px^2+py^2) < 1e-16
        disp("q1 is undefined, infinite solutions");
        q = [];
        return
    end
    disp("We do de same for q1, again having two solutions")
    q1 = atan2(py,px);
    q1p = atan2(-py,-px);
    c1 = cos(q1);
    c1p = cos(q1p);
    s1 = sin(q1);
    s1p= sin(q1p);
    
    disp("We solve a linear system Ax=b for q2, having in total 4 solutions that depends on the combinations of q1 and q3")
    A = [l2+l3*c3  -l3*s3; l3*s3  l2+l3*c3];
    Ap = [l2+l3*c3  -l3*s3p; l3*s3p  l2+l3*c3];
    B = [c1*px+s1*py, pz-d1];
    Bp = [c1p*px+s1p*py, pz-d1];

    syms c2 s2
    eq1 = A*[c2;s2]==B;
    eq2 = Ap*[c2;s2]==B;
    eq3 = A*[c2;s2]==Bp;
    eq4 = Ap*[c2;s2]==Bp;

    [c21,s21] = solve(eq1,[c2,s2]);
    [c22,s22] = solve(eq2,[c2,s2]);
    [c23,s23] = solve(eq3,[c2,s2]);
    [c24,s24] = solve(eq4,[c2,s2]);
        
    q21 = atan2(s21,c21);
    q22 = atan2(s22,c22);
    q23 = atan2(s23,c23);
    q24 = atan2(s24,c24);
    q = [q1 q21 q3; q1p q22 q3; q1 q23 q3p; q1p q24 q3p];
    return
end
end
