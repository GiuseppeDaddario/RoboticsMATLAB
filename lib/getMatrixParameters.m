%Function for compute axis r and angle theta, given a rotation matrix R.
%returns two arrays, one contain the two solution for r and the other
%contains the two solution for theta.
function [r, theta] = getMatrixParameters(R)
%Unicode symbols
sub1 = char(0x2081); %1
sub2 = char(0x2082); %2
sup2 = char(0x00B2); % ^2
sub3 = char(0x2083); %3
subx = char(0x2093); % x
suby = char(0x1D67); % y
subz = char(0x2095); % z
c_theta = char(0x03B8); %theta

if size(R,1) ~= 3 || size(R,2) ~= 3 || abs(det(R) - 1) > 1e-16
        error('R should be a rotation matrix 3x3, with determinant +1');
end
s = (R(1,2)-R(2,1))^2 + (R(1,3)-R(3,1))^2 + (R(2,3)-R(3,2))^2;
Y = sqrt(s);
X = R(1,1) + R(2,2) + R(3,3) - 1;
theta = atan2(Y,X);
fprintf("We compute %c using the atan2 function:\n%c = atan2{±√((R%c%c - R%c%c)%c + (R%c%c - R%c%c)%c + (R%c%c - R%c%c)%c), R%c%c + R%c%c + R%c%c - 1}\n\n",c_theta,c_theta,sub1,sub2,sub2,sub1,sup2,sub1,sub3,sub3,sub1,sup2,sub2,sub3,sub3,sub2,sup2,sub1,sub1,sub2,sub2,sub3,sub3);
if s > 1e-15
    disp("Sin(theta) > 0, so we are in a regular case with two solutions\n\n")
    fprintf("The formula to compute r is:\nr = 1/2sin(%c) * [R%c%c - R%c%c\n\t\t R%c%c - R%c%c\n\t\t R%c%c - R%c%c]\n\n",c_theta,sub3,sub2,sub2,sub3,sub1,sub3,sub3,sub1,sub2,sub1,sub1,sub2);
    r = (1 / (2 * sin(theta))) * [R(3,2) - R(2,3), R(1,3) - R(3,1), R(2,1) - R(1,2)];
else
    disp("Sin(theta) < 0, so we are in a singular case with two solutions\n\n");
    fprintf("The formula to compute r is:\nr = [±√(R%c%c+1)/2\tr%cr%c = R%c%c/2\n     ±√(R%c%c+1)/2  with  r%cr%c = R%c%c/2\n     ±√(R%c%c+1)/2]\tr%cr%c = R%c%c/2\n\n",sub1,sub1,subx,suby,sub1,sub2,sub2,sub2,subx,subz,sub1,sub3,sub3,sub3,suby,subz,sub2,sub3);
    soluzioni_valide = [];
    radici_pos = sqrt((diag(R)+1)/2);
    %cicle for choose the right combination of the sqare roots
    signs = [-1, 1];
    for sx = signs
        for sy = signs
            for sz = signs
                r_candidato = [sx * radici_pos(1), sy * radici_pos(2), sz * radici_pos(3)];
                if abs(r_candidato(1)*r_candidato(2) - R(1,2)/2) < 1e-15 && ...
                        abs(r_candidato(1)*r_candidato(3) - R(1,3)/2) < 1e-15 && ...
                        abs(r_candidato(2)*r_candidato(3) - R(2,3)/2) < 1e-15

                    isDuplicate = false;
                    for i = 1:size(soluzioni_valide, 1)
                        if norm(soluzioni_valide(i, :) - r_candidato) < 1e-15
                            isDuplicate = true;
                            break;
                        end
                    end
                    if ~isDuplicate
                        soluzioni_valide = [soluzioni_valide; r_candidato];
                    end
                end
            end
        end
    end
    r = soluzioni_valide;
end
fprintf("Since we have '±' in the atan2 function, we should consider bot positive and negative values of the angle, and so for the relative axis r\n\n")
r = [r; -r];
theta = [theta; -theta];
end

