function detJ = analysisJB(J)
syms q1 q2 q3 q4 q5 q6
assume([q1,q2,q3,q4,q5,q6],"real")

rankJ = rank(J)
% Normalize the null space
nullJ = simplify(null(J));
for col = 1:size(nullJ, 2)
    % Verifica se il primo elemento della colonna è diverso da zero
    if nullJ(1,col) ~= 0
        % Se il primo elemento è non nullo, normalizza
        nullJ(:,col) = nullJ(:,col) / nullJ(1,col);
    else
        % Se il primo elemento è zero, cerca il primo elemento non nullo
        first_non_zero_idx = find(nullJ(:,col) ~= 0, 1);  % Trova il primo elemento non nullo
        if ~isempty(first_non_zero_idx)
            % Normalizza utilizzando il primo valore non nullo
            nullJ(:,col) = nullJ(:,col) / nullJ(first_non_zero_idx, col);
        else
            % Se tutta la colonna è nulla, usa una piccola costante epsilon
            epsilon = 1e-10;  % Aggiungi una piccola costante per evitare la divisione per zero
            nullJ(1,col) = epsilon;  % Imposta epsilon come primo valore
            nullJ(:,col) = nullJ(:,col) / epsilon;  % Normalizza con epsilon
        end
    end
end

fprintf("Null space (normalized): [dim = %d]\n", size(nullJ, 2));
disp(simplify(nullJ));

RangeJT = orth(J);
for col = 1:size(RangeJT, 2)
    % Verifica se il primo elemento della colonna è diverso da zero
    if RangeJT(1,col) ~= 0
        % Se il primo elemento è non nullo, normalizza
        RangeJT(:,col) = RangeJT(:,col) / RangeJT(1,col);
    else
        % Se il primo elemento è zero, cerca il primo elemento non nullo
        first_non_zero_idx = find(RangeJT(:,col) ~= 0, 1);  % Trova il primo elemento non nullo
        if ~isempty(first_non_zero_idx)
            % Normalizza utilizzando il primo valore non nullo
            RangeJT(:,col) = RangeJT(:,col) / RangeJT(first_non_zero_idx, col);
        else
            % Se tutta la colonna è nulla, usa una piccola costante epsilon
            epsilon = 1e-10;  % Aggiungi una piccola costante per evitare la divisione per zero
            RangeJT(1,col) = epsilon;  % Imposta epsilon come primo valore
            RangeJT(:,col) = RangeJT(:,col) / epsilon;  % Normalizza con epsilon
        end
    end
end
fprintf("Range space: [dim = %d]\n", size(RangeJT, 2));
disp(simplify(RangeJT));

% Controlla che la matrice sia 3x3
if size(J,1) == 3 && size(J,2) == 3
    detJ = sarrus(J);
    eq = detJ == 0;
    % Compute the singularities
    real_vars = intersect(symvar(eq), [q1,q2,q3,q4,q5,q6]);
    sol = solve(eq, real_vars, 'ReturnConditions', true);
    results = struct2cell(sol);
    disp('Singularities when:');
    for i = 1:numel(real_vars)
        fprintf('%s: %s\n', char(real_vars(i)), char(results{i}));
    end
    i = numel(real_vars) + 1;
    fprintf('%s: %s\n', char('params'), char(results{i}));
    fprintf('%s: %s\n', char('conditions'), char(results{i + 1}));
else
    % Remove each column at a time for the analysis
    N = size(J,2);
    dets = cell(1, N);
    eqs = [];
    for i = 1:N
        j = J;
        j(:,i) = [];
        if size(j,1) == 3 && size(j,2) == 3
            dets{i} = sarrus(j);
        elseif size(j,1) ~= size(j,2)
            dets{i} = simplify(det(j * transpose(j)));
        else
            dets{i} = simplify(det(j));
        end
        fprintf("Det of J-%d:\n", i);
        disp(dets{i});
        e = dets{i} == 0;
        eqs = [eqs, e];
    end

    % Compute the singularities
    real_vars = intersect(symvar(eqs), [q1,q2,q3,q4,q5,q6]);
    sol = solve(eqs, real_vars, 'ReturnConditions', true);
    results = struct2cell(sol);
    disp('Singularities when:');
    for i = 1:numel(real_vars)
        fprintf('%s: %s\n', char(real_vars(i)), char(results{i}));
    end
    i = numel(real_vars) + 1;
    fprintf('%s: %s\n', char('params'), char(results{i}));
    fprintf('%s: %s\n', char('conditions'), char(results{i + 1}));
end
end