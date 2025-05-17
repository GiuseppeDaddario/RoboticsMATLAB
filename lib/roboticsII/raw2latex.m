function raw2latex(expr)
%FORMULA2LATEX Converte una formula simbolica in stringa LaTeX leggibile
%
%   latex_str = formula2latex(expr)
%
%   INPUT:
%     expr - Espressione simbolica (ad es. ottenuta con il Symbolic Math Toolbox)
%
%   OUTPUT:
%     latex_str - Stringa LaTeX pronta per essere usata in grafica o esportazione

if ~isa(expr, 'sym')
    error('L''input deve essere un''espressione simbolica (sym).');
end

% Usa la funzione latex di MATLAB
raw_latex = latex(expr);

% Aggiunge i delimitatori display math se desiderato
latex_str = ['$$', raw_latex, '$$'];  % per display mode in figure

figure;
text(0.1, 0.5, latex_str, 'Interpreter', 'latex', 'FontSize', 16);
axis off
end