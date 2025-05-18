function disp_algorithm(w, v, vc, T, i, M, c, g)
    persistent fig ax y_offset lines_history n_total

    is_final = nargin == 8;

    if is_final

        % --- Matrix M ---
        % title
        title_line_str = sprintf('$$\\underline{\\textbf{Inertia Matrix M}}$$');
        h_text = text(ax, 0.02, -y_offset, title_line_str, ...
            'Interpreter', 'latex', 'FontSize', 13, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax);
        M_str = latex(M);
        M_str = apply_dot_notation(M_str);
        M_line_str = sprintf('$$\\mathbf{M(q)} = %s$$', M_str);
        h_text = text(ax, 0.02, -y_offset, M_line_str, ...
            'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax, 1.0);

        % --- Coriolis c ---
        % title
        title_line_str = sprintf('$$\\underline{\\textbf{Coriolis coefficients c}}$$');
        h_text = text(ax, 0.02, -y_offset, title_line_str, ...
            'Interpreter', 'latex', 'FontSize', 13, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax);
        c_cell = arrayfun(@(e) apply_dot_notation(latex(simplify(e, 'Steps', 20))), c, 'UniformOutput', false);
        for i = 1:length(c_cell)
            row_str = sprintf('$$\\mathbf{c}_{%d}(q, \\dot{q}) = %s$$', i, c_cell{i});
            h_text = text(ax, 0.02, -y_offset, row_str, ...
                'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'top');
            lines_history{end+1} = h_text;
            y_offset = y_offset + get_text_height_approx(h_text, ax, 1.0);
        end

        % --- Gravity g ---
        % title
        title_line_str = sprintf('$$\\underline{\\textbf{Gravity coefficients g}}$$');
        h_text = text(ax, 0.02, -y_offset, title_line_str, ...
            'Interpreter', 'latex', 'FontSize', 13, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax);
        g_str = apply_dot_notation(latex(g));
        g_line_str = sprintf('$$\\mathbf{g(q)} = %s$$', g_str);
        h_text = text(ax, 0.02, -y_offset, g_line_str, ...
            'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax, 1.0);

        % Update axes
        ylim(ax, [-y_offset, 0.05]);
        return;
    end

    % --- Init ---
    if isempty(fig) || ~isvalid(fig)
        fig = figure('Name', 'LaTeX Robot Debug', 'NumberTitle', 'off', 'Color', 'w');
        ax = axes('Parent', fig);
        axis(ax, 'off');
        hold(ax, 'on');
        y_offset = 0;
        lines_history = {};
    end

    if i == 1 && ~isempty(lines_history)
        for h_idx = 1:numel(lines_history)
            if isvalid(lines_history{h_idx})
                delete(lines_history{h_idx});
            end
        end
        lines_history = {};
        y_offset = 0;
    end

    simplify_steps = 20;
    to_latex = @(x_in) arrayfun(@(e) apply_dot_notation(latex(simplify(e, 'Steps', simplify_steps))), x_in, 'UniformOutput', false);

    wi = {}; if ~isempty(w{i}), wi = to_latex(w{i}); end
    vi = {}; if ~isempty(v{i}), vi = to_latex(v{i}); end
    vci = {}; if ~isempty(vc{i}), vci = to_latex(vc{i}); end

    n = max([numel(wi), numel(vi), numel(vci)]);

    % Spacing parameters
    section_spacing = 0.02;

    % Step title
    title_line_str = sprintf('$$\\underline{\\textbf{Step %d}}$$', i);
    h_text = text(ax, 0.02, -y_offset, title_line_str, ...
        'Interpreter', 'latex', 'FontSize', 13, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    y_offset = y_offset + get_text_height_approx(h_text, ax);

    % w vector
    h_text = text(ax, 0.02, -y_offset, '$$\mathbf{w}$$', ...
        'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    y_offset = y_offset + get_text_height_approx(h_text, ax);
    for k = 1:n
        str = get_cell_latex(wi, k);
        h_text = text(ax, 0.04, -y_offset, sprintf('$$%s$$', str), ...
            'Interpreter', 'latex', 'FontSize', 11, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax, 0.8);
    end

    % v vector
    h_text = text(ax, 0.02, -y_offset, '$$\mathbf{v}$$', ...
        'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    y_offset = y_offset + get_text_height_approx(h_text, ax);
    for k = 1:n
        str = get_cell_latex(vi, k);
        h_text = text(ax, 0.04, -y_offset, sprintf('$$%s$$', str), ...
            'Interpreter', 'latex', 'FontSize', 11, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax, 0.8);
    end

    % vc vector
    h_text = text(ax, 0.02, -y_offset, '$$\mathbf{v_c}$$', ...
        'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    y_offset = y_offset + get_text_height_approx(h_text, ax);
    for k = 1:n
        str = get_cell_latex(vci, k);
        h_text = text(ax, 0.04, -y_offset, sprintf('$$%s$$', str), ...
            'Interpreter', 'latex', 'FontSize', 11, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        y_offset = y_offset + get_text_height_approx(h_text, ax, 0.8);
    end

    y_offset = y_offset + section_spacing * 2;

    % Kinetic energy
    T_str = apply_dot_notation(latex(simplify(T{i}, 'Steps', simplify_steps)));
    kin_line_str = sprintf('$$T_{%d} = %s$$', i, T_str);
    h_text = text(ax, 0.02, -y_offset, kin_line_str, ...
        'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    y_offset = y_offset + get_text_height_approx(h_text, ax) + section_spacing * 2;

    ylim(ax, [-y_offset, 0.05]);
    drawnow;
end

function str_out = apply_dot_notation(str_in)
    % Remove unnecessary LaTeX formatting
    str_out = strrep(str_in, '\left', '');
    str_out = strrep(str_out, '\right', '');

    % Replace \mathrm{ddq}_i with \ddot{q}_i
    tokens = regexp(str_out, '\\mathrm{ddq}_?(\d*)', 'tokens');
    if ~isempty(tokens)
        for k = 1:numel(tokens)
            idx = tokens{k}{1};
            if isempty(idx)
                repl = '\\ddot{q}';
                str_out = regexprep(str_out, '\\mathrm{ddq}', repl, 1);
            else
                match = ['\\mathrm{ddq}_' idx];
                repl = ['\\ddot{q}_{' idx '}'];
                str_out = strrep(str_out, match, repl);
            end
        end
    end

    % Replace \mathrm{dq}_i with \dot{q}_i
    tokens = regexp(str_out, '\\mathrm{dq}_?(\d*)', 'tokens');
    if ~isempty(tokens)
        for k = 1:numel(tokens)
            idx = tokens{k}{1};
            if isempty(idx)
                repl = '\\dot{q}';
                str_out = regexprep(str_out, '\\mathrm{dq}', repl, 1);
            else
                match = ['\\mathrm{dq}_' idx];
                repl = ['\\dot{q}_{' idx '}'];
                str_out = strrep(str_out, match, repl);
            end
        end
    end
end

function out = add_dotted_notation(dot_type, idx)
    if isempty(idx)
        out = sprintf('\\%s{q}', dot_type);
    else
        out = sprintf('\\%s{q}_{%s}', dot_type, idx);
    end
end

function out_latex = get_cell_latex(cell_array_of_latex, idx)
    if idx <= numel(cell_array_of_latex) && ~isempty(cell_array_of_latex{idx})
        out_latex = cell_array_of_latex{idx};
    else
        out_latex = '\\phantom{0}';
    end
end

function height_approx = get_text_height_approx(text_handle, ax, scale_factor)
    if nargin < 3, scale_factor = 1.0; end
    font_size_pts = get(text_handle, 'FontSize');
    height_approx = (font_size_pts / 100) * 0.25 * scale_factor;
    try
        extent = get(text_handle, 'Extent');
        height_approx = extent(4);
    catch
        height_approx = (font_size_pts / 100) * 0.25 * scale_factor;
    end
    height_approx = height_approx * scale_factor;
end