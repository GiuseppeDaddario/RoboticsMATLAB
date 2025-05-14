function [tau, Y, pi_coeffs] = regression_matrix(M, c, g, pi_coeffs)
%REGRESSION_MATRIX Computes torque tau and regression matrix Y
% Inputs:
%   M - Mass matrix (symbolic or numeric)
%   c - Coriolis/centrifugal vector
%   g - Gravity vector
%   pi_coeffs (optional) - Inertial parameter vector
% Outputs:
%   tau - Torque vector
%   Y - Regression matrix (Jacobian of tau wrt pi_coeffs)
%   pi_coeffs - Inertial parameter vector used

    n_links = size(M, 1);  % Number of joints/links

    % Define symbolic variables
    ddq_vector = sym('ddq', [n_links, 1], 'real');  % Joint accelerations

    % Inertial parameters
    m       = sym('m', [n_links, 1], 'real');
    m_rc_x  = sym('m_rc_x', [n_links, 1], 'real');
    m_rc_y  = sym('m_rc_y', [n_links, 1], 'real');  % Fixed typo here
    m_rc_z  = sym('m_rc_z', [n_links, 1], 'real');
    I_xx    = sym('I_xx', [n_links, 1], 'real');
    I_xy    = sym('I_xy', [n_links, 1], 'real');
    I_xz    = sym('I_xz', [n_links, 1], 'real');
    I_yy    = sym('I_yy', [n_links, 1], 'real');
    I_yz    = sym('I_yz', [n_links, 1], 'real');
    I_zz    = sym('I_zz', [n_links, 1], 'real');

    % Initialize parameter vector if not provided
    if nargin < 4 || isempty(pi_coeffs)
        pi_coeffs = [];
        for i = 1:n_links
            pi_coeffs = [pi_coeffs;
                         m(i);
                         m_rc_x(i);
                         m_rc_y(i);
                         m_rc_z(i);
                         I_xx(i);
                         I_xy(i);
                         I_xz(i);
                         I_yy(i);
                         I_yz(i);
                         I_zz(i)];
        end
    end

    % Compute torque and regression matrix
    tau = simplify(M * ddq_vector + c + g);
    Y = simplify(jacobian(tau, pi_coeffs));

    % Optional: remove zero columns (commented out)
    % Y(:, all(Y == 0, 1)) = [];

end