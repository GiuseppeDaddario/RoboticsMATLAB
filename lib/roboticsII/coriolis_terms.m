%% function for computing the coriolis terms of the robot's dynamic model
%% input variables:
% - M: inertia matrix of the robot
%% output variables:
% - c: coriolis terms of the robot's dynamic model

function c = coriolis_terms(M)
n_links = size(M,1);
q_vector = sym('q', [n_links,1],'real');
dq_vector = sym('dq', [n_links,1],'real');
q = num2cell(q_vector);
dq = num2cell(dq_vector);
c = cell(1,n_links);
for n=1:n_links
    term_1 = jacobian(M(:,n),q_vector);
    term_3 = diff(M,[q{n}]);
    c{n} = 1/2 * (term_1+term_1.'-term_3);
    for i=1:size(c{n},1)
        for j=1:size(c{n},2)
            c{n}(i,j) = simplify(c{n}(i,j)*dq{i}*dq{j});
        end
    end
    c{n} = sum(c{n}(:));
end
c = [c{:}].';
end
