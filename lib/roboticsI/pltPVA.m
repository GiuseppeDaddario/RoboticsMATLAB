function pltPVA(eqs, T_in,T_f)
    % Assicuriamoci che eqs sia un cell array unidimensionale
    if ~iscell(eqs)
        error('L''input deve essere un cell array.');
    end
    
    % Se eqs è una matrice (es. 3x3), lo trasformiamo in un vettore
    eqs = eqs(:);  % Converte una matrice in un vettore colonna

    % Controllo che il numero di equazioni sia valido (multiplo di 2 o 3)
    num_eqs = numel(eqs);
    if mod(num_eqs, 3) == 0
        % Se è multiplo di 3, ci sono anche accelerazioni
        num_joints = num_eqs / 3;
        has_accelerations = true;
    elseif mod(num_eqs, 2) == 0
        % Se è multiplo di 2, ci sono solo posizioni e velocità
        num_joints = num_eqs / 2;
        has_accelerations = false;
    else
        error('Il numero di equazioni deve essere un multiplo di 2 o 3.');
    end

    % Definizione delle variabili simboliche
    syms t T

    % Creazione di un vettore di valori per s (da 0 a T con 100 punti)
    s_values = linspace(T_in, T_f, 100);

    % Calcolo dei valori numerici delle equazioni
    eqs_values = cellfun(@(eq) double(subs(eq, {t,T}, {s_values,T_f})), eqs, 'UniformOutput', false);

    % Grafico delle Posizioni (Prime num_joints equazioni)
    figure;
    hold on;
    colors = lines(num_joints);  % Genera colori distinti
    for i = 1:num_joints
        plot(s_values, eqs_values{i}, 'Color', colors(i, :), 'DisplayName', sprintf('Joint %d - Position', i));
    end
    hold off;
    xlabel('s');
    ylabel('Posizioni articolazioni');
    legend show;
    title('Joint Positions');

    % Grafico delle Velocità (Equazioni da num_joints+1 a 2*num_joints)
    figure;
    hold on;
    for i = 1:num_joints
        plot(s_values, eqs_values{i + num_joints}, 'Color', colors(i, :), 'DisplayName', sprintf('Joint %d - Velocity', i));
    end
    hold off;
    xlabel('s');
    ylabel('Velocità articolazioni');
    legend show;
    title('Joint Velocities');

    % Se ci sono accelerazioni, grafico delle Accelerazioni (Equazioni da 2*num_joints+1 a 3*num_joints)
    if has_accelerations
        figure;
        hold on;
        for i = 1:num_joints
            plot(s_values, eqs_values{i + 2*num_joints}, 'Color', colors(i, :), 'DisplayName', sprintf('Joint %d - Acceleration', i));
        end
        hold off;
        xlabel('s');
        ylabel('Accelerazioni articolazioni');
        legend show;
        title('Joint Accelerations');
    end
end