function [spline1, spline2] = build_spline(qa_dot_0, qb_dot_1, qs, qm, qg, T)
    syms tau_a tau_b vm1 vm2 c_a1_1 c_a2_1 c_a3_1 c_b1_1 c_b2_1 c_b3_1 ...
                       c_a1_2 c_a2_2 c_a3_2 c_b1_2 c_b2_2 c_b3_2
    
    % Vettori delle configurazioni iniziale, media e finale
    qs1 = qs(1);
    qs2 = qs(2);
    qm1 = qm(1);
    qm2 = qm(2);
    qg1 = qg(1);
    qg2 = qg(2);

    % Definizione delle equazioni per q1
    eq1_1 = c_a1_1 == qa_dot_0;
    eq2_1 = c_a2_1 + c_a3_1 == qm1 - qs1;
    eq3_1 = 4*c_a2_1 + 6*c_a3_1 == vm1*T;
    eq4_1 = c_b1_1 == qb_dot_1;
    eq5_1 = c_b2_1 - c_b3_1 == qm1 - qg1;
    eq6_1 = -4*c_b2_1 + 6*c_b3_1 == vm1*T;
    eq7_1 = c_a2_1 + 3*c_a3_1 == c_b2_1 - 3*c_b3_1;

    % Definizione delle equazioni per q2
    eq1_2 = c_a1_2 == qa_dot_0;
    eq2_2 = c_a2_2 + c_a3_2 == qm2 - qs2;
    eq3_2 = 4*c_a2_2 + 6*c_a3_2 == vm2*T;
    eq4_2 = c_b1_2 == qb_dot_1;
    eq5_2 = c_b2_2 - c_b3_2 == qm2 - qg2;
    eq6_2 = -4*c_b2_2 + 6*c_b3_2 == vm2*T;
    eq7_2 = c_a2_2 + 3*c_a3_2 == c_b2_2 - 3*c_b3_2;

    % Risoluzione separata per q1 e q2
    sol1 = solve([eq1_1, eq2_1, eq3_1, eq4_1, eq5_1, eq6_1, eq7_1], ...
                 [vm1, c_a1_1, c_a2_1, c_a3_1, c_b1_1, c_b2_1, c_b3_1]);
    
    sol2 = solve([eq1_2, eq2_2, eq3_2, eq4_2, eq5_2, eq6_2, eq7_2], ...
                 [vm2, c_a1_2, c_a2_2, c_a3_2, c_b1_2, c_b2_2, c_b3_2]);

    % Definizione delle spline
    spline1 = [qs1 + sol1.c_a1_1*tau_a + sol1.c_a2_1*tau_a^2 + sol1.c_a3_1*tau_a^3;
               qs2 + sol2.c_a1_2*tau_a + sol2.c_a2_2*tau_a^2 + sol2.c_a3_2*tau_a^3];

    spline2 = [qg1 + sol1.c_b1_1*(tau_b-1) + sol1.c_b2_1*(tau_b-1)^2 + sol1.c_b3_1*(tau_b-1)^3;
               qg2 + sol2.c_b1_2*(tau_b-1) + sol2.c_b2_2*(tau_b-1)^2 + sol2.c_b3_2*(tau_b-1)^3];

    % Semplificazione finale
    spline1 = vpa(collect(spline1,tau_a), 4);
    spline2 = vpa(collect(spline2,tau_b), 4);
end