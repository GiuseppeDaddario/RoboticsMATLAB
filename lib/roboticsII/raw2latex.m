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
            warning('Detected unsupported LaTeX structures. Using element-by-element rendering.');
            xlim(ax, [0, cols + 1]);
            ylim(ax, [-rows - 1, 0]);
            for i = 1:rows
                for j = 1:cols
                    entry = latex(expr(i,j));
                    entry = strrep(entry, '\\', '');
                    entry = replace_dots(entry);
                    text(ax, j, -i, ['$', entry, '$'], ...
                        'Interpreter', 'latex', ...
                        'FontSize', FONTSIZE, 'HorizontalAlignment', 'left');
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