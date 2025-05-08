function delta_ro = calibration(table,parameters,q_values,measurements)
n_links = size(table,1);
[~,T,~,~] = DK(table); clc
f = T(1:3,4);

columns = [];

a = sym('a', [1,n_links], 'real');
alpha = sym('alpha', [1,n_links], 'real');
d = sym('d', [1,n_links], 'real');
q = sym('q', [1,n_links], 'real');

dfda = jacobian(f,a);
dfdalpha = jacobian(f,alpha);
dfdd = jacobian(f,d);
dfdq = jacobian(f,q);

n_measurements = size(measurements,1);
phi=sym(zeros(n_measurements,4*n_links));

for i = 1:n_measurements
    row_idx = (i-1)*3 + 1;
    phi(row_idx:row_idx+2, :) = vpa(subs([dfdalpha dfda dfdd dfdq], q, q_values(i,:)),4);
end


if any(strcmp(parameters, 'alpha'))
    columns = [columns, 1:n_links];
end
if any(strcmp(parameters, 'a'))
    columns = [columns, (n_links+1):(2*n_links)];
end
if any(strcmp(parameters, 'd'))
    columns = [columns, (2*n_links+1):(3*n_links)];
end
if any(strcmp(parameters, 'q'))
    columns = [columns, (3*n_links+1):(4*n_links)];
end

phi = phi(:, columns);

delta_r = reshape(measurements.', [], 1);
delta_ro = vpa(pinv(phi)*delta_r,4);
end