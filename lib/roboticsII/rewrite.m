%% function to display the elements of the inertia matrix so to simply collect coefficients by inspection
function new_M = rewrite(M)
rows = size(M,1);
cols = size(M,2);
new_M = M;
elements = [];
q_vector = sym('q', [rows,1], 'real');
for i=1:rows
    for j=1:cols
        m = collect(collect(M(i,j),cos(q_vector)),sin(q_vector));
        varname = sym(sprintf('m%d%d', i, j)); % crea variabile simbolica m11, m12, ecc.
        eq = varname == m;                     % costruisce equazione m11 == ...
        fprintf('%s = %s;\n', char(varname), char(m));
        new_M(i,j) = m;
        elements = [elements; eq]; %% if needed, return them
    end
end
end

%% extension: chatGpt. trying to compute the coefficients 
% % Espressioni simboliche da usare come base
% function baseFuncs = generaFunzioniBase(q)
%     % q: array simbolico degli angoli (es. [q1 q2 q3])
%     baseFuncs = sym(1); % include costante 1
%     n = length(q);
% 
%     % Aggiungi sin(qi) e cos(qi)
%     for i = 1:n
%         baseFuncs(end+1) = sin(q(i));
%         baseFuncs(end+1) = cos(q(i));
%     end
% 
%     % Aggiungi sin(qi + qj) e cos(qi + qj) con i < j per evitare ripetizioni
%     for i = 1:n
%         for j = i+1:n
%             baseFuncs(end+1) = sin(q(i) + q(j));
%             baseFuncs(end+1) = cos(q(i) + q(j));
%         end
%     end
% end
% 
% fprintf('Espressioni con coefficienti raccolti:\n\n');
% for i = 1:4
%     for j = 1:4
%         expr = M(i,j);
%         if expr ~= 0
%             coeffs = zeros(1, length(baseFuncs));
%             for k = 1:length(baseFuncs)
%                 coeffs(k) = simplify( ...
%                     feval(symengine, 'coeff', expr, baseFuncs(k)) );
%             end
%             fprintf('m%d%d = ', i, j);
%             terms = [];
%             for k = 1:length(baseFuncs)
%                 if coeffs(k) ~= 0
%                     terms = [terms, sprintf('(%s)*%s + ', char(coeffs(k)), char(baseFuncs(k)))];
%                 end
%             end
%             if isempty(terms)
%                 fprintf('0\n');
%             else
%                 terms = erase(terms, ' + ');
%                 fprintf('%s\n', terms);
%             end
%         end
%     end
% end