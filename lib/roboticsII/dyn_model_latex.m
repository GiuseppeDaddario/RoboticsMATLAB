function dyn_model_latex(varargin)
% --- Costanti di Configurazione ---
FONTSIZE_NORMAL = 20;
FONTSIZE_TITLE = 21;
X_MARGIN = 0.02;         % Margine orizzontale sinistro
Y_TOP_MARGIN = 0.98;     % Punto di partenza verticale (torna al valore originale)
LINE_SPACING = 2.6;      % Fattore di spaziatura per linee normali
SECTION_SPACING = 3;   % Fattore di spaziatura dopo un titolo di sezione
FORMULA_SPACING = 1.2;
H_GAP = 0.01;            % Spazio orizzontale tra le equazioni w, v, vc

% --- Variabili Persistenti per mantenere lo stato della figura ---
persistent fig ax current_y lines_history

% --- Gestione delle modalità di chiamata (inizializzazione, step, finale) ---
is_init = false;
is_final = false;

    switch nargin
        case 0
            is_init = true;
        case 2
            is_final = true;
            M = varargin{1};
            c = varargin{1};
            g = sym([0;0;0]);
        case 3
            is_final = true;
            M = varargin{1};
            c = varargin{2};
            g = varargin{3};
        case 5
            w = varargin{1}; 
            v = varargin{2};
            vc = varargin{3};
            T = varargin{4}; 
            i = varargin{5};
    end


% --- Inizializzazione della Figura (solo alla prima chiamata) ---
if isempty(fig) || ~isvalid(fig) || is_init
    % Se è una chiamata di inizializzazione, chiudi la vecchia figura se esiste
    if ~isempty(fig) && isvalid(fig)
        close(fig);
    end
    
    fig = figure('Name', 'Moving Frames Algorithm', 'NumberTitle', 'off', 'Color', 'w', ...
                 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0 0 1 1]);
    axis(ax, [0 1 0 1]); % Usa un sistema di coordinate fisso e normalizzato
    axis(ax, 'off');
    hold(ax, 'on');
    
    % Resetta lo stato per una nuova esecuzione
    current_y = Y_TOP_MARGIN;
    lines_history = {};
end

% Se la chiamata è solo per inizializzazione, mostra le formule e termina
if is_init
    % Titolo principale
    h_text = text(ax, X_MARGIN, current_y, '$$\underline{\textbf{Steps formulas}}$$','Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;

    % Lista delle formule
    w_formula = "^i\!w_i = ^{i-1}\!R_i^T \left(^{i-1}\!w_{i-1} + (1 - \sigma_i) {}^{i-1}\!z_{i-1} \right)";
    vi_formula = "^i\!v_i = ^{i-1}\!R_i^T  \left(^{i-1}\!v_i + \sigma_i \dot{q}_i \cdot {}^{i-1}\!z_{i-1} \right) + \left(^{i-1}\!w_i \times {}^{i-1}\!r_{i-1,i} \right)";
    vc_formula = "v_{ci} = v_i + \left(w_i \times r_{ci}\right)";
    T_formula = "T_i = \frac{1}{2} m_i \left( v_{ci}^T v_{ci} \right) + \frac{1}{2} w_i^T  I_{ci} w_i";

    formula_list = {w_formula, vi_formula, vc_formula, T_formula};
    current_x = X_MARGIN;
    
    for k = 1:numel(formula_list)
        h_text = text(ax, current_x, current_y, sprintf('$$%s$$', formula_list{k}), ...
            'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        drawnow;

        % Aggiorna la posizione x in base alla larghezza effettiva del testo
        current_x = current_x + get_text_width(h_text) * FORMULA_SPACING;
    end
    
    % Aggiorna la posizione y dopo le formule
    current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;
    
    % Non restringere la vista - lascia che il contenuto si estenda naturalmente
    return;
end

% --- Logica di Stampa per gli Step intermedi ---
if ~is_final
    simplify_steps = 20;

    % Titolo dello Step
    title_line_str = sprintf('$$\\underline{\\textbf{Step %d}}$$', i);
    h_title = text(ax, X_MARGIN, current_y, title_line_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_title;
    drawnow;
    current_y = current_y - get_effective_text_height(h_title) * LINE_SPACING;

    % Stampa di w, v, vc sulla stessa linea
    w_str = ['$$\mathbf{w} = ', apply_dot_notation(latex(simplify(w{i}, 'Steps', simplify_steps))), '$$'];
    v_str = ['$$\mathbf{v} = ', apply_dot_notation(latex(simplify(v{i}, 'Steps', simplify_steps))), '$$'];
    vc_str = ['$$\mathbf{v_c} = ', apply_dot_notation(latex(simplify(vc{i}, 'Steps', simplify_steps))), '$$'];
    
    % Crea gli oggetti di testo (inizialmente invisibili) per misurarne la larghezza
    h_w = text(ax, 0, current_y, w_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top', 'Visible', 'off');
    h_v = text(ax, 0, current_y, v_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top', 'Visible', 'off');
    h_vc = text(ax, 0, current_y, vc_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top', 'Visible', 'off');
    lines_history = [lines_history, {h_w, h_v, h_vc}];
    drawnow;

    % Ottieni le dimensioni (altezza e larghezza) in unità normalizzate
    ext_w = get(h_w, 'Extent');
    ext_v = get(h_v, 'Extent');
    ext_vc = get(h_vc, 'Extent');

    % Posiziona gli oggetti di testo orizzontalmente e rendili visibili
    x_current = X_MARGIN;
    set(h_w, 'Position', [x_current, current_y, 0], 'Visible', 'on');
    x_current = x_current + ext_w(3) + H_GAP;
    set(h_v, 'Position', [x_current, current_y, 0], 'Visible', 'on');
    x_current = x_current + ext_v(3) + H_GAP;
    set(h_vc, 'Position', [x_current, current_y, 0], 'Visible', 'on');

    % Calcola l'altezza effettiva massima della riga e aggiorna la posizione verticale
    max_line_height = max([get_effective_text_height(h_w), get_effective_text_height(h_v), get_effective_text_height(h_vc)]);
    current_y = current_y - max_line_height * LINE_SPACING;

    % Stampa dell'energia cinetica T
    T_str = apply_dot_notation(latex(simplify(T{i}, 'Steps', simplify_steps)));
    kin_line_str = sprintf('$$T_{%d} = %s$$', i, T_str);
    h_kin = text(ax, X_MARGIN, current_y, kin_line_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_kin;
    drawnow;
    current_y = current_y - get_effective_text_height(h_kin) * SECTION_SPACING;

% --- Logica di Stampa Finale (M, c, g) ---
else 
    % --- Matrice di Inerzia M ---
    
    h_text = text(ax, X_MARGIN, current_y, '$$\underline{\textbf{Inertia matrix M}}$$','Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;

    % --- Formula M ---
    M_formula = "$$T = \frac{1}{2} \dot{\mathbf{q}}^\top \mathbf{M} \dot{\mathbf{q}}$$";
    h_text = text(ax, X_MARGIN, current_y, M_formula,'Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;

    M_str = apply_dot_notation(latex(M));
    M_line_str = sprintf('$$\\mathbf{M(q)} = %s$$', M_str);
    h_text = text(ax, X_MARGIN, current_y, M_line_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * SECTION_SPACING;

    % --- Vettore di Coriolis c ---
    h_text = text(ax, X_MARGIN, current_y, '$$\underline{\textbf{Coriolis coefficients c}}$$','Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;

    % --- Formula C ---
    C_formula = "$$C_i(q) = \frac{1}{2} \left( \frac{\partial \mathbf{M}_{i}}{\partial \mathbf{q}} + \left( \frac{\partial \mathbf{M}_{i}}{\partial \mathbf{q}} \right)^\top - \frac{\partial \mathbf{M}}{\partial q_i} \right)$$";
    c_formula = "$$c_i(q, \dot{\mathbf{q}}) = \dot{\mathbf{q}}^\top C_i(q) \, \dot{\mathbf{q}}$$";
    current_x = X_MARGIN;
    h_text = text(ax, current_x, current_y, C_formula, ...
        'Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    text_width = get_text_width(h_text);
    current_x = current_x + text_width + 0.02;
    h_text = text(ax, current_x, current_y, c_formula, ...
        'Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * (LINE_SPACING+1.5);
    
    c_cell = arrayfun(@(e) apply_dot_notation(latex(simplify(e, 'Steps', 20))), c, 'UniformOutput', false);
    for idx = 1:length(c_cell)
        row_str = sprintf('$$\\mathbf{c}_{%d}(q, \\dot{q}) = %s$$', idx, c_cell{idx});
        h_text = text(ax, X_MARGIN, current_y, row_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top');
        lines_history{end+1} = h_text;
        drawnow;
        current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;
    end
    % Aggiungi uno spazio extra dopo l'ultima riga di c
    current_y = current_y - get_effective_text_height(h_text) * (SECTION_SPACING - LINE_SPACING);

    % --- Vettore di Gravità g ---
    h_text = text(ax, X_MARGIN, current_y, '$$\underline{\textbf{Gravity vector g}}$$','Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;

    % --- Formula G ---
    U_formula = "$$U_i = -m_i \, \mathbf{g}^\top \mathbf{r}_{0,ci}$$";
    G_formula = '$$g(q) = \left( \frac{\partial \mathbf{U}_{\mathrm{tot}}}{\partial \mathbf{q}} \right)^\top$$';
    current_x = X_MARGIN;
    h_text = text(ax, current_x, current_y, U_formula, ...
        'Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    text_width = get_text_width(h_text);
    current_x = current_x + text_width + 0.02;
    h_text = text(ax, current_x, current_y, G_formula, ...
        'Interpreter', 'latex', 'FontSize', FONTSIZE_TITLE, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * LINE_SPACING;

    g_str = apply_dot_notation(latex(g));
    g_line_str = sprintf('$$\\mathbf{g(q)} = %s$$', g_str);
    h_text = text(ax, X_MARGIN, current_y, g_line_str, 'Interpreter', 'latex', 'FontSize', FONTSIZE_NORMAL, 'VerticalAlignment', 'top');
    lines_history{end+1} = h_text;
    drawnow;
    current_y = current_y - get_effective_text_height(h_text) * SECTION_SPACING;
end

% --- Aggiornamento finale della vista ---
% Mantieni la vista fissa, permetti lo scroll naturale
drawnow;

end

% --- Funzioni Helper ---

function height = get_effective_text_height(text_handle)
    % Restituisce l'altezza effettiva del testo, riducendo lo spazio bianco
    try
        extent = get(text_handle, 'Extent');
        raw_height = extent(4);
        
        % Applica un fattore di correzione per ridurre lo spazio bianco
        % Il fattore 0.7 è empirico e può essere aggiustato
        height = raw_height * 0.7;
        
        % Imposta un'altezza minima per evitare spaziature troppo piccole
        min_height = 0.02;
        height = max(height, min_height);
        
    catch
        % Fallback nel caso improbabile che Extent non sia disponibile
        font_size_pts = get(text_handle, 'FontSize');
        height = font_size_pts / 72 * 0.025; % Stima più conservativa
    end
end

function height = get_text_height(text_handle)
    % Restituisce l'altezza di un oggetto di testo in unità normalizzate.
    try
        extent = get(text_handle, 'Extent');
        height = extent(4); % L'altezza è il quarto elemento di Extent
    catch
        % Fallback nel caso improbabile che Extent non sia disponibile
        font_size_pts = get(text_handle, 'FontSize');
        height = font_size_pts / 72 * 0.035; % Stima approssimativa dell'altezza normalizzata
    end
end

function h = get_text_width(h_text)
    extent = get(h_text, 'Extent');
    h = extent(3);  % larghezza
end

function str_out = apply_dot_notation(str_in)
    % Questa funzione converte la notazione testuale (es. dq1) in notazione LaTeX con punti (es. \dot{q}_1)
    str_out = strrep(str_in, '\left', '');
    str_out = strrep(str_out, '\right', '');

    % Sostituisce ddq_1, ddq_2, etc. con \ddot{q}_{1}, \ddot{q}_{2}
    tokens = regexp(str_out, 'ddq_\{?(\d+)\}?', 'tokens');
    if ~isempty(tokens)
        for k = 1:numel(tokens)
            idx = tokens{k}{1};
            str_out = regexprep(str_out, ['ddq_\{?' idx '\}?'], ['\\ddot{q}_{' idx '}'], 1);
        end
    end
    str_out = regexprep(str_out, 'ddq(?!_)', '\\ddot{q}'); % Per ddq senza indice

    % Sostituisce dq_1, dq_2, etc. con \dot{q}_{1}, \dot{q}_{2}
    tokens = regexp(str_out, 'dq_\{?(\d+)\}?', 'tokens');
    if ~isempty(tokens)
        for k = 1:numel(tokens)
            idx = tokens{k}{1};
            str_out = regexprep(str_out, ['dq_\{?' idx '\}?'], ['\\dot{q}_{' idx '}'], 1);
        end
    end
    str_out = regexprep(str_out, 'dq(?!_)', '\\dot{q}'); % Per dq senza indice
end