function [minimal, rank_Y] = is_minimal(Y, a, joint_syms)
% Verifica se la parametrizzazione a è minima data la matrice Y
% Input:
%   Y           - matrice simbolica di regressione
%   a           - vettore dei parametri simbolici
%   joint_syms  - simboli delle variabili (es. [q1 dq1 ddq1 q2 dq2 ddq2])
%
% Output:
%   minimal     - true se la parametrizzazione è minima
%   rank_Y      - rango numerico della matrice estesa Y

    % Numero di test casuali
    N = 50;
    
    % Numero di righe di Y e colonne (parametri)
    [~, num_params] = size(Y);
    
    % Valori numerici da usare nei test
    Y_full = [];
    
    for i = 1:N
        % Genera valori numerici casuali realistici tra -1 e 1
        random_values = 2*rand(1, length(joint_syms)) - 1;
        
        % Sostituisci questi valori nella matrice Y
        Y_i = vpa(subs(Y, joint_syms, random_values),2);
        
        % Accumula nella matrice totale
        Y_full = [Y_full; Y_i];
    end

    % Calcola il rango numerico della matrice totale
    rank_Y = rank(Y_full);  % la soglia aiuta con la stabilità numerica

    % Se il rango è uguale al numero di parametri, la parametrizzazione è minima
    minimal = (rank_Y == num_params);
    
    % Stampa risultato se chiamata senza output
    if nargout == 0
        if minimal
            fprintf("✅ It's a MINIMAL parametrization (rank = %d / %d)\n", rank_Y, num_params);
        else
            fprintf("❌ It's NON a minimal parametrization (rank = %d / %d)\n", rank_Y, num_params);
        end
    end
end