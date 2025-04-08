function [sol, coeff_sym, spA, spB, spC, spA_dot, spB_dot, spC_dot, spA_ddot, spB_ddot, spC_ddot] = spline3(v_in, v_fin, p1, p2, p3, p4, t1, t2, t3, t4)
    syms t c_a0 c_a1 c_a2 c_a3 c_b0 c_b1 c_b2 c_b3 c_c0 c_c1 c_c2 c_c3
    
    % Correzione nella definizione di tau
    tau_a = (t - t1) / (t2 - t1);
    tau_b = (t - t2) / (t3 - t2);
    tau_c = (t - t3) / (t4 - t3);

    % Definizione delle spline cubiche
    spA = c_a0 + c_a1*tau_a + c_a2*tau_a^2 + c_a3*tau_a^3;
    spB = c_b0 + c_b1*tau_b + c_b2*tau_b^2 + c_b3*tau_b^3;
    spC = c_c0 + c_c1*tau_c + c_c2*tau_c^2 + c_c3*tau_c^3;

    % Calcolo delle derivate
    spA_dot = diff(spA, t);
    spA_ddot = diff(spA_dot, t);
    spB_dot = diff(spB, t);
    spB_ddot = diff(spB_dot, t);
    spC_dot = diff(spC, t);
    spC_ddot = diff(spC_dot, t);

    % Equazioni di continuità
    eq1 = c_a0 == p1;
    eq2 = c_b0 == p2;
    eq3 = c_c0 == p3;
    eq4 = simplify(subs(spA, t, t2)) == p2;
    eq5 = simplify(subs(spB, t, t3)) == p3;
    eq6 = simplify(subs(spC, t, t4)) == p4;
    eq7 = simplify(subs(spA_dot, t, t1)) == v_in;
    eq8 = simplify(subs(spA_dot, t, t2)) == simplify(subs(spB_dot, t, t2));
    eq9 = simplify(subs(spB_dot, t, t3)) == simplify(subs(spC_dot, t, t3));
    eq10 = simplify(subs(spA_ddot, t, t2)) == simplify(subs(spB_ddot, t, t2));
    eq11 = simplify(subs(spB_ddot, t, t3)) == simplify(subs(spC_ddot, t, t3));
    eq12 = simplify(subs(spC_dot, t, t4)) == v_fin;

    % Risoluzione del sistema simbolico
    sol = solve([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12], ...
                [c_a0, c_a1, c_a2, c_a3, c_b0, c_b1, c_b2, c_b3, c_c0, c_c1, c_c2, c_c3]);

    % Memorizzare i coefficienti simbolici (espressioni complete)
    coeff_sym = struct();
    coeff_sym.c_a0 = sol.c_a0;
    coeff_sym.c_a1 = sol.c_a1;
    coeff_sym.c_a2 = sol.c_a2;
    coeff_sym.c_a3 = sol.c_a3;
    coeff_sym.c_b0 = sol.c_b0;
    coeff_sym.c_b1 = sol.c_b1;
    coeff_sym.c_b2 = sol.c_b2;
    coeff_sym.c_b3 = sol.c_b3;
    coeff_sym.c_c0 = sol.c_c0;
    coeff_sym.c_c1 = sol.c_c1;
    coeff_sym.c_c2 = sol.c_c2;
    coeff_sym.c_c3 = sol.c_c3;

    % Sostituzione delle soluzioni nelle spline
    spA = subs(spA, fieldnames(sol), struct2cell(sol));
    spB = subs(spB, fieldnames(sol), struct2cell(sol));
    spC = subs(spC, fieldnames(sol), struct2cell(sol));

    spA_dot = subs(spA_dot, fieldnames(sol), struct2cell(sol));
    spB_dot = subs(spB_dot, fieldnames(sol), struct2cell(sol));
    spC_dot = subs(spC_dot, fieldnames(sol), struct2cell(sol));

    spA_ddot = subs(spA_ddot, fieldnames(sol), struct2cell(sol));
    spB_ddot = subs(spB_ddot, fieldnames(sol), struct2cell(sol));
    spC_ddot = subs(spC_ddot, fieldnames(sol), struct2cell(sol));

    % Semplificazione finale
    spA = vpa(collect(spA, tau_a), 4);
    spB = vpa(collect(spB, tau_b), 4);
    spC = vpa(collect(spC, tau_c), 4);
    spA_dot = vpa(collect(spA_dot, tau_a), 4);
    spB_dot = vpa(collect(spB_dot, tau_b), 4);
    spC_dot = vpa(collect(spC_dot, tau_c), 4);
    spA_ddot = vpa(collect(spA_ddot, tau_a), 4);
    spB_ddot = vpa(collect(spB_ddot, tau_b), 4);
    spC_ddot = vpa(collect(spC_ddot, tau_c), 4);
end