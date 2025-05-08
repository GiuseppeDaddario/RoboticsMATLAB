function [tau,Y,pi_coeffs] = regression_matrix(M,c,g,pi_coeffs)
n_links = size(M,1);

% Syms definitions
ddq_vector = sym('ddq', [n_links,1], 'real');
m = sym('m', [n_links,1], 'real');
m_rc_x=sym('m_rc_x', [n_links,1], 'real'); m_rc_y=sym('m_r_cy', [n_links,1], 'real'); m_rc_z=sym('m_rc_z', [n_links,1], 'real');
I_xx=sym('I_xx', [n_links,1], 'real'); I_xy=sym('I_xy', [n_links,1], 'real'); I_xz=sym('I_xz', [n_links,1], 'real'); I_yy=sym('I_yy', [n_links,1], 'real'); I_yz=sym('I_yz', [n_links,1], 'real'); I_zz=sym('I_zz', [n_links,1], 'real');
if isempty(pi_coeffs)
    for i=1:n_links
        pi_coeffs = [pi_coeffs; m(i); m_rc_x(i);m_rc_y(i);m_rc_z(i); I_xx(i);I_xy(i);I_xz(i);I_yy(i);I_yz(i);I_zz(i)];
    end
end
tau = simplify(M*ddq_vector+c+g);
Y =  simplify(jacobian(tau,pi_coeffs'));
% Y(:, all(Y == 0, 1)) = [];
end