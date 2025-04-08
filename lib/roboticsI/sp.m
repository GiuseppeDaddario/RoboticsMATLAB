function [S1, S2] = build_spline(qa_dot_0, qb_dot_1, qs, qm, qg, T)
    % Definisci i nodi (tempi)
    tau = [0, 0.5, 1];  % esempio: inizio, punto medio, fine
    
    % I valori delle configurazioni
    q1_vals = [qs(1), qm(1), qg(1)];  % Configurazione per il giunto 1
    q2_vals = [qs(2), qm(2), qg(2)];  % Configurazione per il giunto 2
    
    % Calcola h_i (distanze tra i nodi)
    h1 = tau(2) - tau(1);
    h2 = tau(3) - tau(2);

    % Calcola le differenze finite (b_i)
    b1 = (q1_vals(2) - q1_vals(1)) / h1;
    b2 = (q1_vals(3) - q1_vals(2)) / h2;
    
    % Sistema tridiagonale per calcolare le derivate seconde M
    A = [2*(h1 + h2), h2; h2, 2*(h1 + h2)];
    B = [3*(b1 + b2); 3*(b2 - b1)];

    % Risoluzione del sistema tridiagonale
    M = A \ B;

    % Calcolo dei coefficienti delle spline per il giunto 1
    a1 = (M(2) - M(1)) / (3*h1);
    a2 = M(1);
    a3 = b1 - h1*(2*M(1) + M(2)) / 3;
    a4 = q1_vals(1);
    
    % Spline per il giunto 1 (S1)
    S1 = @(t) a1*(t - tau(1)).^3 + a2*(t - tau(1)).^2 + a3*(t - tau(1)) + a4;

    % Ripeti lo stesso per il giunto 2
    b1 = (q2_vals(2) - q2_vals(1)) / h1;
    b2 = (q2_vals(3) - q2_vals(2)) / h2;
    A = [2*(h1 + h2), h2; h2, 2*(h1 + h2)];
    B = [3*(b1 + b2); 3*(b2 - b1)];
    M = A \ B;
    
    % Calcolo dei coefficienti delle spline per il giunto 2
    a1 = (M(2) - M(1)) / (3*h1);
    a2 = M(1);
    a3 = b1 - h1*(2*M(1) + M(2)) / 3;
    a4 = q2_vals(1);
    
    % Spline per il giunto 2 (S2)
    S2 = @(t) a1*(t - tau(1)).^3 + a2*(t - tau(1)).^2 + a3*(t - tau(1)) + a4;
end