function [t1,t1v,t1a] = cubic(mode,conditions, T,customMode)
    syms tau s
    
    % Estrarre posizioni e velocità
    phi_i = conditions(1);    % Posizione iniziale
    phi_f = conditions(2);    % Posizione finale
    phi_dot_i = conditions(3);    % Velocità iniziale
    phi_dot_f = conditions(4);    % Velocità finale
    
    % Delta della posizione
    delta_phi = phi_f - phi_i;
    if mode == "cubicTime"
    % Formula per la traiettoria cubica in funzione di tau e T
        t1 = phi_i + ((phi_dot_i * T) * tau) + (((3 * delta_phi) - ((phi_dot_f + 2 * phi_dot_i) * T)) * tau^2) + (((-2 * delta_phi) + ((phi_dot_f + phi_dot_i) * T)) * tau^3);
    % Derivata della traiettoria (velocità angolare)
        syms t
        t1t = subs(t1,tau,t/T);
        t1v = diff(t1t, t);
        t1a = diff(t1v,t);
    elseif mode == "cubicTimeCustom"
    % Formula per la traiettoria cubica in funzione di tau e T
        t1 = phi_i + (phi_dot_i * s) + (((3 * delta_phi) - ((phi_dot_f + 2 * phi_dot_i))) * s^2) + (((-2 * delta_phi) + ((phi_dot_f + phi_dot_i))) * s^3);
        t1 = subs(t1,s,customMode);
    % Derivata della traiettoria (velocità angolare)
        syms t
        t1v = diff(t1, t);
        t1a = diff(t1v,t);
    else
    % Formula per la traiettoria cubica in funzione di s
        t1 = phi_i + (phi_dot_i * s) + (((3 * delta_phi) - ((phi_dot_f + 2 * phi_dot_i))) * s^2) + (((-2 * delta_phi) + ((phi_dot_f + phi_dot_i))) * s^3);
    % Derivata della traiettoria (velocità angolare)
        t1v = diff(t1, s);
    end
    
    
    % Approssimazione con precisione a 5 decimali
    t1 = vpa(t1, 4);
    t1v = vpa(t1v, 4);
    t1a = vpa(t1a, 4);
end