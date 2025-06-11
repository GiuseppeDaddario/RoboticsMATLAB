function [dyn,lambda,dA,dD,E,F] = reduced_dynamics(M,c,g,A,D,plot)
if nargin == 5
    plot = false;
    returna = false;
elseif nargin == 0
    plot = true;
    returna = true;
end

if plot
    dyn_eq = "(F^T M F) \dot{v} = F^T (u - c - g + M (E \dot{A} + F \dot{D}) F v) ";
    lambda_eq = "\lambda = E^T (M F \dot{v} - M (E \dot{A} + F \dot{D}) F v + c + g - u) ";
    
    figure;
    ax = axes;
    axis(ax, 'off');
    hold(ax, 'on');
    title(ax, "Reduced Dynamics formulas", 'Interpreter', 'latex', 'FontSize', 20);
    
    xlim(ax, [-10,10]);
    ylim(ax, [-10,10]);

    FONTSIZE = 16;

    text(ax, -12, 8.5, sprintf('$$%s$$', dyn_eq), ...
            'Interpreter', 'latex', 'FontSize', FONTSIZE, "HorizontalAlignment","left");
    text(ax, -12, 6.5, sprintf('$$%s$$', lambda_eq), ...
            'Interpreter', 'latex', 'FontSize', FONTSIZE, "HorizontalAlignment","left");

    hold(ax, 'off');

    if returna
        return;
    end
end

syms u real
n_links = size(M,1);
q = sym('q', [n_links,1], 'real');
dq = sym('dq', [n_links,1], 'real');

AD = [A; D];
dA=diff_wrt(A,q,1);
if nargin == 4
    D = [0, A(1)^-1];

    d = det(AD);
    D = D / d;
end
dD = diff_wrt(D,q,1);
AD_T = AD^-1;
E = AD_T(:,1);
F = AD_T(:,2);
v = D*dq; dv=diff_wrt(v,dq,2);

dyn = simplify((F.'*M*F)*dv == F.'*(u-c-g+M*(E*dA+F*dD)*F*v));
lambda = simplify(E.'*(M*F*dv-M*(E*dA+F*dD)*F*v+c+g-u));

end