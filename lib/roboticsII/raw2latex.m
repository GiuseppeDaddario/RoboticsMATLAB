function raw2latex(expr,eq_title,unsupported)
%RAW2LATEX Display symbolic expression(s) as LaTeX text with dq → q̇, ddq → q̈.
%   - Vectors: displayed line by line.
%   - Matrices: rendered fully if LaTeX is simple.
%   - Fallback to element-by-element if LaTeX contains unsupported structures.

if nargin == 1
    unsupported = false;
    eq_title = "";
elseif nargin == 2
    unsupported = false;
end

FONTSIZE = 20;
if ~isa(expr, 'sym')
    error('Input must be symbolic.');
end

figure;
ax = axes;
axis(ax, 'off');
hold(ax, 'on');
title(ax, eq_title, 'Interpreter', 'latex', 'FontSize', 20);

[rows, cols] = size(expr);
xlim(ax, [-10,10]);
ylim(ax, [-10,10]);

% Helper function to replace dq/ddq with proper dots in LaTeX string
    function outStr = replace_dots(latexStr)
        outStr = regexprep(latexStr, 'ddq', '\\ddot{q}');
        outStr = regexprep(outStr, '(?<!d)dq', '\\dot{q}');
    end

if isvector(expr)
    raw_latex = arrayfun(@latex, expr(:), 'UniformOutput', false);
    for i = 1:length(raw_latex)
        clean_latex = replace_dots(raw_latex{i});
        txt = ['$', clean_latex, '$'];
        text(ax, 0.5, -i, txt, 'Interpreter', 'latex', ...
            'FontSize', FONTSIZE, 'HorizontalAlignment', 'left');
    end

else
    full_latex = latex(expr);

    if unsupported
        xlim(ax, [0, cols + 1]);
        ylim(ax, [-rows - 1, 0]);
        counter = 1;
        for i = 1:rows
            for j = 1:cols
                % Crea nome simbolico m_ij
                varname = sym(sprintf('m%d%d', i, j));
                % Crea equazione simbolica
                eq = varname == expr(i,j);
                % Convertila in LaTeX
                eq_latex = latex(eq);
                eq_latex = replace_dots(eq_latex);
                % Stampa in figura
                text(ax, 0, -0.3*counter, ['$', eq_latex, '$'], ...
                    'Interpreter', 'latex', ...
                    'FontSize', FONTSIZE, 'HorizontalAlignment', 'left');
                counter = counter + 1;
            end
        end
    else
        clean_latex = replace_dots(full_latex);
        text(ax, 0.05, 0.5, ['$', clean_latex, '$'], ...
            'Interpreter', 'latex', 'FontSize', FONTSIZE);
    end
end

hold(ax, 'off');
end