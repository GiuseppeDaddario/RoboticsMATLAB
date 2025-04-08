function [s_max_abs, f_max_abs, s_min_abs, f_min_abs] = findMax(f, s, min_val, max_val)
    % f: funzione da analizzare
    % s: variabile simbolica
    % min_val: limite inferiore dell'intervallo
    % max_val: limite superiore dell'intervallo

    % Calcola la derivata prima della funzione
    df = diff(f, s); 

    % Trova i punti critici risolvendo df = 0
    critical_s = double(solve(df, s)); 

    % Filtra i punti critici che sono nell'intervallo [min_val, max_val]
    critical_s = critical_s(critical_s >= min_val & critical_s <= max_val); 

    % Calcola i valori della funzione nei punti critici
    critical_values = double(subs(f, s, critical_s));

    % Considera anche i valori agli estremi dell'intervallo
    f_min = double(subs(f, s, min_val));
    f_max = double(subs(f, s, max_val));

    % Aggiungi i valori agli estremi dell'intervallo come candidati
    critical_s = critical_s(:); % Forza critical_s a essere un vettore colonna
    critical_s = [min_val; max_val; critical_s]; % Concatenazione verticale (colonna)
    critical_values = [f_min; f_max; critical_values(:)]; % Concatenazione verticale (colonna)

    % Trova il massimo in valore assoluto
    [~, idx_max] = max(abs(critical_values));
    s_max_abs = critical_s(idx_max);
    f_max_abs = critical_values(idx_max);

    % Trova il minimo in valore assoluto
    [~, idx_min] = min(abs(critical_values));
    s_min_abs = critical_s(idx_min);
    f_min_abs = critical_values(idx_min);

    % Mostra i risultati
    fprintf("Il massimo in valore assoluto è in s = %.4f con valore f(s) = %.4f\n", s_max_abs, f_max_abs);
    fprintf("Il minimo in valore assoluto è in s = %.4f con valore f(s) = %.4f\n", s_min_abs, f_min_abs);
end