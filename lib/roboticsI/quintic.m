function [t1, t1v, t1a] = quintic(conditions,T)
    syms t
    
    % Estrarre posizioni e velocità
    phi_i = conditions(1);    % Posizione iniziale
    phi_f = conditions(2);    % Posizione finale
    phi_dot_i = conditions(3);    % Velocità iniziale
    phi_dot_f = conditions(4);    % Velocità finale
    phi_ddot_i = conditions(5);
    phi_ddot_f = conditions(6);
    
    % Delta della posizione
    delta_phi = phi_f - phi_i;
    
    % Formula per la traiettoria cubica in funzione di t
    t1 = phi_i + (delta_phi * (6*t^5 - 15*t^4 + 10*t^3)) + (phi_dot_f*T*(-4*t^3 + 7*t^4 - 3*t^5));
    
    % Derivata della traiettoria (velocità angolare)
    t1v = diff(t1, t);
    t1a = diff(t1v,t);
    
    % Approssimazione con precisione a 5 decimali
    t1 = vpa(t1, 5);
    t1v = vpa(t1v, 5);
    t1a = vpa(t1a,5);
end