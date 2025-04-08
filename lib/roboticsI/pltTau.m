function eqs_transformed = pltTau(mode,eqs, T,customMode)
    if nargin < 4
        customMode = [];
    end
    % Assicuriamoci che eqs sia un cell array unidimensionale
    if ~iscell(eqs)
        error('L''input deve essere un cell array.');
    end
    
    % Se eqs è una matrice (es. 3x3), lo trasformiamo in un vettore
    eqs = eqs(:);  % Converte una matrice in un vettore colonna

    % Controllo che il numero di equazioni sia pari
    num_eqs = numel(eqs);
    if mod(num_eqs, 2) ~= 0
        error('Il numero di equazioni deve essere pari (ogni joint ha posizione e velocità).');
    end

    % Numero di giunti (ogni giunto ha una posizione e una velocità)
    num_joints = num_eqs / 2;

    % Definizione delle variabili simboliche
    syms t tau s

    if mode == "time"
    % Sostituzione t -> s/T nelle equazioni
        eqs_transformed = cellfun(@(eq) subs(eq, tau, t/T), eqs, 'UniformOutput', false);
    elseif mode == "customTime"
        eqs_transformed = cellfun(@(eq) subs(eq, s, customMode), eqs, 'UniformOutput', false);
        eqs_transformed = cellfun(@(eq) subs(eq, tau, t/T), eqs_transformed, 'UniformOutput', false);
        for i = 1:num_joints
            eqs_transformed{i+num_joints} = diff(eqs_transformed{i},t);
        end
    elseif mode == "space"
        eqs_transformed = cellfun(@(eq) subs(eq, s, t), eqs, 'UniformOutput', false);
    end
  
    % Creazione di un vettore di valori per s (da 0 a 2 con 100 punti)
    t_values = linspace(0, T, 100);

    % Calcolo dei valori numerici delle equazioni
    eqs_values = cellfun(@(eq) double(subs(eq, t, t_values)), eqs_transformed, 'UniformOutput', false);
    
    % Grafico delle Posizioni (Prime num_joints equazioni)
    figure;
    hold on;
    colors = lines(num_joints);  % Genera colori distinti
    for i = 1:num_joints
        plot(t_values, eqs_values{i}, 'Color', colors(i, :), 'DisplayName', sprintf('Joint %d', i));
    end
    hold off;
    if mode == "space"
        xlabel('s');
    else
      xlabel('t');  
    end
    ylabel('Posizioni articolazioni');
    legend show;
    title('Joint Positions');

    % Grafico delle Velocità (Ultime num_joints equazioni)
    figure;
    hold on;
    for i = 1:num_joints
        plot(t_values, eqs_values{i + num_joints}, 'Color', colors(i, :), 'DisplayName', sprintf('Joint %d', i));
    end
    hold off;
    if mode == "space"
        xlabel('s');
    else
      xlabel('t');  
    end
    ylabel('Velocità articolazioni');
    legend show;
    title('Joint Velocities');
end