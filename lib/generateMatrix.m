%Function for generating a rotation matrix R given an axis r and an angle
%theta. Returns the rotation matrix. If the angle theta is defined as a symbol,
%it returns the general formula to compute the rotation matrix around the
%axis
function R = generateMatrix(r, theta)
% Unicode codes for symbols
sup2 = char(0x00B2); % 2
subx = char(0x2093); % x
suby = char(0x1D67); % y
subz = char(0x2095); % z
c_theta = char(0x03B8); % theta
supT = char(0x1D40); % T

rx = r(1);
ry = r(2);
rz = r(3);
norm_r = sqrt(rx^2 + ry^2 + rz^2);

%If theta is a symbol, compute the general rotation matrix
if isa(theta, 'sym')
    sth = sin(theta);
    cth = cos(theta);
    dcth = 1 - cth;
    R = [rx^2*dcth+cth, rx*ry*dcth-rz*sth, rx*rz*dcth+ry*sth;
         rx*ry*dcth+rz*sth, ry^2*dcth+cth, ry*rz*dcth-rx*sth;
         rx*rz*dcth-ry*sth, ry*rz*dcth+rx*sth, rz^2*dcth+cth];
    R = simplify(R);
    fprintf("The general rotation matrix around the vector r of a given angle %s is:\n\n",theta);
    disp(R)
    return
else
    
    % Normalize
    if norm_r > 0
        rx = rx / norm_r;
        ry = ry / norm_r;
        rz = rz / norm_r;
    end

    epsilon = 1e-10; 
    if abs(theta) < epsilon
        fprintf("Since theta is exactly, or very close, to 0 I build the identity matrix:\n");
        R = eye(3)
        return
    elseif abs(theta - pi) < epsilon
        dcth = 2; % 1 - (-1)
        fprintf("Since theta is exactly, or very close, to pi I build the matrix in a stable way:\n");
        fprintf("R = [(2r%c%c)-1    2r%cr%c     2r%cr%c\n", subx, sup2, subx, suby, subx, subz);
        fprintf("      2r%cr%c     (2r%c%c)-1   2r%cr%c\n", subx, suby, suby, sup2, suby, subz);
        fprintf("      2r%cr%c      2r%cr%c    (2r%c%c)-1]\n", subx, subz, suby, subz, subz, sup2);
        R = [rx^2*dcth-1, rx*ry*dcth, rx*rz*dcth;
             rx*ry*dcth, ry^2*dcth-1, ry*rz*dcth;
             rx*rz*dcth, ry*rz*dcth, rz^2*dcth-1];
        return
    else
        sth = sin(theta);
        cth = cos(theta);
        dcth = 1 - cth;
        
        fprintf('The rotation matrix is obtained from the formula:\nR(%s,r) = rr%s + (I - rr%s)c(%s) + S(r)s(%s)\n', c_theta, supT, supT, c_theta, c_theta);
        R = [rx^2*dcth+cth, rx*ry*dcth-rz*sth, rx*rz*dcth+ry*sth;
             rx*ry*dcth+rz*sth, ry^2*dcth+cth, ry*rz*dcth-rx*sth;
             rx*rz*dcth-ry*sth, ry*rz*dcth+rx*sth, rz^2*dcth+cth];
    end
end
end