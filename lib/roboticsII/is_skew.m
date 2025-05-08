function res_bool = is_skew(M)    
    % Verifica che M sia quadrata
    [rows, cols] = size(M);
    if rows ~= cols
        error('La matrice deve essere quadrata.');
    end
    
    % Controlla la condizione di skew-simmetria
    res = all( M + M.' == 0, 'all' );
    if res == 1
        res_bool = 'True';
    else
        res_bool = 'False';
    end
end