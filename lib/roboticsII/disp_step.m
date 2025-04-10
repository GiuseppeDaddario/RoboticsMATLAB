% function built with chatGPT

function disp_step(w,v,vc,T,i)
disp('========================================================================================================================')
fprintf('                                                 Step %d\n', i)
disp('========================================================================================================================')

% Semplifica e converti in celle di stringhe
wi = arrayfun(@char, simplify(w{i}, 'Steps', 5), 'UniformOutput', false);
vi = arrayfun(@char, simplify(v{i}, 'Steps', 5), 'UniformOutput', false);
vci = arrayfun(@char, simplify(vc{i}, 'Steps', 5), 'UniformOutput', false);

% Calcola padding massimo
pad = @(x) max(cellfun(@strlength, x));
w_pad = pad(wi); v_pad = pad(vi); vc_pad = pad(vci);

% Header
fprintf('%-*s | %-*s | %-*s\n', w_pad, 'w', v_pad, 'v', vc_pad, 'vc');
fprintf('%s\n', repmat('-', 1, w_pad + v_pad + vc_pad + 6));

% Riga per riga
n = max([numel(wi), numel(vi), numel(vci)]);
for k = 1:n
    fprintf('%-*s | %-*s | %-*s\n', ...
        w_pad, iif(k <= numel(wi), wi{k}, ''), ...
        v_pad, iif(k <= numel(vi), vi{k}, ''), ...
        vc_pad, iif(k <= numel(vci), vci{k}, ''));
end

% Energia cinetica
fprintf('\nT{%d} = %s\n', i, char(simplify(T{i}, 'Steps', 5)));
disp('------------------------------------------------------------------------------------------------------------------------')
end

function out = iif(cond, val, fallback)
if cond, out = val; else, out = fallback; end
end