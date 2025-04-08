% Function for computing the inverse kinematics with the iterative Newton method
% Takes parameters:
% r -> matrix of the direct kinematics solution
% q, l -> symbolic vars in r [e.g. q=[q1,q2], l=[l1,l2]]
% q_val, l_val -> initial values of the vars in r [e.g. q=[pi,pi], l=[1,1]]
% p -> vector of the target point
% tol -> tolerance of convergence
% Returns the configuration of the joints
function theta = newtonMethod(r, q, l, q_val, l_val, p, max_iter, tol)
% Init
FK = matlabFunction(r, 'Vars', [q, l]);
all_vals = num2cell([q_val, l_val]);
theta = q_val(:);
converged = false;
error_history = zeros(1, max_iter);

for k = 1:max_iter
    % Compute the direct kinemtics
    x_current = FK(all_vals{:});
    disp("x_current = "), disp(x_current);

    % Compute the error
    error = p(:) - x_current(:);
    error_history(k) = norm(error)


    % Check convergence
    if norm(error) < tol
        converged = true;
        fprintf("Converged in %d iterations\nq = \n", k);
        disp(theta)
        break;
    end

    % Compute the Jacobian
    J = jacobian(r, q);
    J_current = subs(J, [q, l], [theta.', l_val]);
    J_current = double(J_current);

    % Update theta
    delta_theta = pinv(J_current) * error;
    theta = theta + delta_theta
    all_vals = num2cell([theta.', l_val]);
end

if ~converged
    disp("Newton method didn't converge");
end
error_history = error_history(1:k);
pltError(error_history);

end