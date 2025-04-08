%% function for renaming the inertia matrix coefficients with a_i's coefficients (e.g. a1,a2,a3,...)
%% input variables:
% - M: original inertia matrix
%% output variables:
% - M: renamed inertia matrix

function M = rename_coefficients(M)
n_links = size(M,1);
q_vector = sym('q', [n_links,1],'real');
k = 0;
all_coeffs = containers.Map();
for i=1:size(M,1)
    for j = 1:size(M,2)
        [coefficients,terms] = coeffs(M(i,j),[cos(q_vector),sin(q_vector)]);
        for o=1:size(terms,2)
            term_str = string(terms(o));
            if isKey(all_coeffs, term_str)
                all_coeffs(term_str) = [all_coeffs(term_str), coefficients(o)];
            else
                all_coeffs(term_str) = coefficients(o);
            end
        end
        k = k+1;
    end
end
% extracting the greatest common divisor among coefficients
values_array = values(all_coeffs);
gcd_array = sym(zeros(1, length(values_array)));
for n=1:size(values_array,2)
    gcd_value = values_array{n}(1);
    for o=2:length(values_array{n})
        gcd_value = gcd(gcd_value, values_array{n}(o));
    end
    gcd_array(n) = gcd_value;
end

%substituting coefficients
tosub = [];
for i = 1:numel(values_array)
    current_values = values_array{i};
    current_values = current_values(:);
    tosub = [tosub; current_values];
end
tosub = flip(unique(tosub));
idx = [];
for i=1: size(gcd_array,2)
    if gcd_array(i)~=1
        fprintf("Substituting %s with a%d\n",gcd_array(i),i);
        idx =[idx,i];
        M = subs(expand(M),gcd_array(i),sym(sprintf('a%d', i),'real'));
    end
end
k=1;
for i=1:size(tosub,1)
    while ismember(k,idx)
        k = k+1;
    end
    M_new = subs(expand(M),tosub(i),sym(sprintf('a%d', k),'real'));
    if ~isequal(M, M_new)
        fprintf("Substituting %s with a%d\n",tosub(i),k);
        M = M_new;
        idx = [idx,k];
    else
        k = k-1;
    end
end
end