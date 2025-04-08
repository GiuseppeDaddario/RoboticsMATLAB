function pltSplines(spA, spB, spC, spA_dot, spB_dot, spC_dot, spA_ddot, spB_ddot, spC_ddot, t1, t2, t3, t4)
    syms t

    % Definizione degli intervalli di tempo
    t_values_A = linspace(t1, t2, 100);
    t_values_B = linspace(t2, t3, 100);
    t_values_C = linspace(t3, t4, 100);
    
    % Valutazione delle spline nei rispettivi intervalli
    pos_A = double(subs(spA, t, t_values_A));
    pos_B = double(subs(spB, t, t_values_B));
    pos_C = double(subs(spC, t, t_values_C));

    vel_A = double(subs(spA_dot, t, t_values_A));
    vel_B = double(subs(spB_dot, t, t_values_B));
    vel_C = double(subs(spC_dot, t, t_values_C));

    acc_A = double(subs(spA_ddot, t, t_values_A));
    acc_B = double(subs(spB_ddot, t, t_values_B));
    acc_C = double(subs(spC_ddot, t, t_values_C));

    % Plot Posizione
    figure;
    subplot(3,1,1);
    hold on;
    plot(t_values_A, pos_A, 'r', 'LineWidth', 2);
    plot(t_values_B, pos_B, 'g', 'LineWidth', 2);
    plot(t_values_C, pos_C, 'b', 'LineWidth', 2);
    title('Posizione');
    xlabel('Tempo');
    ylabel('Posizione');
    legend('spA', 'spB', 'spC');
    grid on;
    hold off;

    % Plot Velocità
    subplot(3,1,2);
    hold on;
    plot(t_values_A, vel_A, 'r', 'LineWidth', 2);
    plot(t_values_B, vel_B, 'g', 'LineWidth', 2);
    plot(t_values_C, vel_C, 'b', 'LineWidth', 2);
    title('Velocità');
    xlabel('Tempo');
    ylabel('Velocità');
    legend('spA_dot', 'spB_dot', 'spC_dot');
    grid on;
    hold off;

    % Plot Accelerazione
    subplot(3,1,3);
    hold on;
    plot(t_values_A, acc_A, 'r', 'LineWidth', 2);
    plot(t_values_B, acc_B, 'g', 'LineWidth', 2);
    plot(t_values_C, acc_C, 'b', 'LineWidth', 2);
    title('Accelerazione');
    xlabel('Tempo');
    ylabel('Accelerazione');
    legend('spA_ddot', 'spB_ddot', 'spC_ddot');
    grid on;
    hold off;
end