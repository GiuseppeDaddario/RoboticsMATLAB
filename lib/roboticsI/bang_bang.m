function [T,Ta] = bang_bang(s, V, A)
    % Calcolo del tempo minimo per Bang Bang o Bang Coast Bang (rest-to-rest)
    % Input:
    % s = distanza
    % V = velocità massima
    % A = accelerazione massima
    
    % Calcolo dei tempi
    Ta_1 = sqrt(s / A);  % Tempo accelerazione se non c'è coast (Bang Bang)
    Ta_2 = V / A;        % Tempo accelerazione se c'è coast (Bang Coast Bang)
    T_1 = 2 * Ta_1;      % Tempo totale se non c'è coast
    T_2 = (s * A + V^2) / (A * V); % Tempo totale se c'è coast
    
    % Controllo del tipo di movimento
    if s <= V^2 / A
        % Se la distanza è minore o uguale alla distanza per raggiungere la velocità massima
        T = T_1; % Non c'è coast, solo accelerazione e decelerazione
        Ta = Ta_2;
        if s < V^2 / A
            disp("distance s is less than (V^2 / A)")
        else
            disp("distance s is equal to (V^2 / A)")
        end
        disp("Ta computed as V/A")
        disp("T total computed as 2 * Ta")
        motion_type = 'Bang Bang (No Coast)';
    else
        % Se la distanza è maggiore della distanza per raggiungere la velocità massima
        T = T_2; % Movimento con coast
        Ta = Ta_1;
        disp("distance s is greater than (V^2 / A)")
        disp("Ta computed as sqrt(s/A)")
        disp("T total computed as (s * A + V^2) / (A * V)")
        motion_type = 'Bang Coast Bang';
    end
    
    % Visualizzazione dei grafici
    figure;
    
    % Creazione dei grafici
    t = linspace(0, T, 1000); % Intervallo temporale
    
    if strcmp(motion_type, 'Bang Bang (No Coast)')
        % Calcolo delle posizioni, velocità e accelerazioni per il caso senza coast (rest-to-rest)
        % Fase di accelerazione
        t_accel = t(t <= Ta_1);
        x_accel = 0.5 * A * t_accel.^2;  % Posizione durante accelerazione
        v_accel = A * t_accel;           % Velocità durante accelerazione
        a_accel = A * ones(size(t_accel)); % Accelerazione costante
        
        % Fase di decelerazione
        t_decel = t(t > Ta_1);  % Decelerazione inizia subito dopo accelerazione
        x_decel = s - 0.5 * A * (T - t_decel).^2; % Posizione durante decelerazione
        v_decel = V - A * (t_decel - Ta_1);      % Velocità durante decelerazione (da V a 0)
        a_decel = -A * ones(size(t_decel));      % Decelerazione costante
        
        % Grafico della posizione
        subplot(3,1,1);
        plot(t_accel, x_accel, 'r', 'LineWidth', 2);
        hold on;
        plot(t_decel, x_decel, 'b', 'LineWidth', 2);
        title('Bang Bang (No Coast) - Posizione');
        xlabel('Tempo [s]');
        ylabel('Posizione [m]');
        legend('Accelerazione', 'Decelerazione');
        
        % Grafico della velocità
        subplot(3,1,2);
        plot(t_accel, v_accel, 'g', 'LineWidth', 2);
        hold on;
        plot(t_decel, v_decel, 'm', 'LineWidth', 2);
        title('Bang Bang (No Coast) - Velocità');
        xlabel('Tempo [s]');
        ylabel('Velocità [m/s]');
        legend('Accelerazione', 'Decelerazione');
        
        % Grafico dell'accelerazione
        subplot(3,1,3);
        plot(t_accel, a_accel, 'b', 'LineWidth', 2);
        hold on;
        plot(t_decel, a_decel, 'k', 'LineWidth', 2);
        title('Bang Bang (No Coast) - Accelerazione');
        xlabel('Tempo [s]');
        ylabel('Accelerazione [m/s^2]');
        legend('Accelerazione', 'Decelerazione');
        
    else
        % Calcolo delle posizioni, velocità e accelerazioni per il caso con coast (rest-to-rest)
        % Fase di accelerazione
        t_accel = t(t <= Ta_2);
        x_accel = 0.5 * A * t_accel.^2;  % Posizione durante accelerazione
        v_accel = A * t_accel;           % Velocità durante accelerazione
        a_accel = A * ones(size(t_accel)); % Accelerazione costante
        
        % Fase di coast
        t_coast = t(t > Ta_2 & t <= T - Ta_2);
        x_coast = 0.5 * A * Ta_2^2 + V * (t_coast - Ta_2);  % Posizione durante coast
        v_coast = V * ones(size(t_coast));  % Velocità costante durante coast
        a_coast = zeros(size(t_coast));    % Accelerazione nulla durante coast
        
        % Fase di decelerazione
        t_decel = t(t > T - Ta_2);
        x_decel = s - 0.5 * A * (T - t_decel).^2; % Posizione durante decelerazione
        v_decel = V - A * (t_decel - (T - Ta_2)); % Velocità durante decelerazione
        a_decel = -A * ones(size(t_decel));      % Decelerazione costante
        
        % Grafico della posizione
        subplot(3,1,1);
        plot(t_accel, x_accel, 'r', 'LineWidth', 2);
        hold on;
        plot(t_coast, x_coast, 'b', 'LineWidth', 2);
        plot(t_decel, x_decel, 'g', 'LineWidth', 2);
        title('Bang Coast Bang - Posizione');
        xlabel('Tempo [s]');
        ylabel('Posizione [m]');
        legend('Accelerazione', 'Coast', 'Decelerazione');
        
        % Grafico della velocità
        subplot(3,1,2);
        plot(t_accel, v_accel, 'g', 'LineWidth', 2);
        hold on;
        plot(t_coast, v_coast, 'm', 'LineWidth', 2);
        plot(t_decel, v_decel, 'k', 'LineWidth', 2);
        title('Bang Coast Bang - Velocità');
        xlabel('Tempo [s]');
        ylabel('Velocità [m/s]');
        legend('Accelerazione', 'Coast', 'Decelerazione');
        
        % Grafico dell'accelerazione
        subplot(3,1,3);
        plot(t_accel, a_accel, 'b', 'LineWidth', 2);
        hold on;
        plot(t_coast, a_coast, 'k', 'LineWidth', 2);
        plot(t_decel, a_decel, 'r', 'LineWidth', 2);
        title('Bang Coast Bang - Accelerazione');
        xlabel('Tempo [s]');
        ylabel('Accelerazione [m/s^2]');
        legend('Accelerazione', 'Coast', 'Decelerazione');
    end
    
    grid on;
end