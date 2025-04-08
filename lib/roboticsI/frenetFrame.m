function [t,n,b] = frenetFrame(p,s)
    p_prime = diff(p,s);
    t = simplify(p_prime/ norm(p_prime));
    t_prime = diff(t,s);
    n = simplify(t_prime/norm(t_prime));
    b = simplify(cross(t, n));
end