function detJ = sarrus(J)
% Controlla che la matrice sia 3x3
if size(J,1) == 3 && size(J,2) == 3
    A = J;

    % Estrai gli elementi della matrice
    a = A(1,1); b = A(1,2); c = A(1,3);
    d = A(2,1); e = A(2,2); f = A(2,3);
    g = A(3,1); h = A(3,2); i = A(3,3);

    % Applica la regola di Sarrus
    detA = (a*e*i + b*f*g + c*d*h) - (c*e*g + b*d*i + a*f*h);
    detJ = simplify(detA);
end