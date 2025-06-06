function res = extract_coeffs(expr, test)
res = sym(zeros(1, numel(test)));

% Trova tutti i sottotermini dell’espressione
    subterms = feval(symengine, 'indets', expr);
    q_terms = sym([]);

    for i = 1:length(subterms)
        term = subterms(i);
        vars_in_term = symvar(term);  % Trova le variabili in quel termine

        % Se almeno una variabile comincia con 'q', aggiungila
        for v = vars_in_term
            if startsWith(char(v), 'q')
                q_terms(end+1) = term;
                break;  % Passa al prossimo termine
            end
        end
    end
    q_terms = unique(q_terms);


stand_coeffs = [cos(q_terms),cos(sum(q_terms)),sin(q_terms),sin(sum(q_terms))];

for i=1:size(test,2)
    for j=1:size(stand_coeffs,2)
        [c, t] = coeffs(expr,stand_coeffs(j));
        idx = find(t == test(i));
        res(i) = res(i) + sum(c(idx));
    end
end
raw2latex([str2sym('Expression'),expr,str2sym('Coefficients'),res])
end