function u = newton_euler_mio(table, joint_types, CoMi,g0)
joint_types = char(joint_types);
n_links = length(joint_types);

% Syms for the derivatives
dq_vector = sym('dq', [n_links,1], 'real');
dq = num2cell(dq_vector);  % dq{1}, dq{2}, ...
ddq_vector = sym('ddq', [n_links,1], 'real');
ddq = num2cell(ddq_vector);  % ddq{1}, ddq{2}, ...

% Sigma (0 = R, 1 = P)
sigma = num2cell(double(joint_types == 'P'));

% Syms for the masses
m = cell(1, n_links);
for i = 1:n_links
    m{i} = sym(sprintf('m%d', i), 'real');
end

% Syms for the inertia
Ic = cell(1, n_links);
for i = 1:n_links
    Ixx = sym(sprintf('Ic%d_xx', i), 'real');
    Iyy = sym(sprintf('Ic%d_yy', i), 'real');
    Izz = sym(sprintf('Ic%d_zz', i), 'real');
    Ic{i} = diag([Ixx, Iyy, Izz]);
end

% Building center of masses from CoMi
rc = cell(1, n_links);
for i = 1:n_links
    rc{i} = CoMi(3*(i-1)+1 : 3*i);
end

% Direct Kinematics
[~, ~, ~, A] = DK(table); clc
R = cell(1, n_links);
p = cell(1, n_links);
r_i = cell(1, n_links);
g_i = cell(1, n_links);
for i = 1:n_links
    R{i} = A{i}(1:3,1:3);
    p{i} = A{i}(1:3,4);
    r_i{i} = R{i}.' * p{i};
    g_i{i} = R{i}.'*g0; %gravity term
end

% Init for the first iteration
w = cell(1, n_links); v = cell(1, n_links); vc = cell(1, n_links); dw = cell(1, n_links); a = cell(1, n_links); a_c = cell(1, n_links);
w_prev = sym([0;0;0]); dw_prev = sym([0;0;0]); v_prev = sym([0;0;0]); a_prev = sym([0;0;0]);
f_next = sym([0;0;0]); tau_next = sym([0;0;0]);
f = cell(1, n_links);
tau = cell(1, n_links);
u = cell(1, n_links);


% ---- forward pass ------
for i = 1:n_links
    w{i}=R{i}.'*(w_prev+dq{i}*[0;0;1]);
    w{i}=vpa(simplify(w{i}));
    fprintf("the w of link %d is:",i);
    disp(w{i})

    % This can also be seen as the derivative of the angular velocity "w" with respect to time.
    dw{i}=R{i}.'*(dw_prev+ddq{i}*[0;0;1]+cross(dq{i}*w_prev,[0;0;1]));
    fprintf("the w_dot of link %d is:",i);
    disp(w_dot)

    a{i}=R{i}.'*a_prev+cross(dw{i},r_i{i})+cross(w{i},r_i{:,i});
    a{i}=vpa(simplify(a{i}));
    fprintf("the a of link %d is:",i);
    disp(a{i});

    a_c{i}=a{i}+cross(dw{i},r_i{i})+cross(w{i},cross(w{i},r_i{i}));
    fprintf("the a_c of link %d is:",i);
    disp(a_c{i});

    %update the prev
    w_prev=w{i};
    dw_prev=w_dot{i};
    a_prev=a{i};
end

% ----- backward pass -------
for i=1:n_links
    ma = m{i}*(a_c{i}-g_i(:,i));
    if i == n_links
        f{i}= ma;
    else
        f{i}=R{i}*f_next+ma;
    end
    f{i}=vpa(simplify(f{i}));
    fprintf("the f of link %d is:",i);
    disp(f{i});

    m_term = cross(f{i},(r_i{i}+rc{i}))+Ic{i}*dw{i}+cross(w{i},(Ic{i}*w{i}));
    if i==n_links
        tau{i}=-m_term;
    else
        tau{i}=R{i}*tau_next+ cross((R{i}*f_next),rc{i})-m_term;
    end
    tau{i}=vpa(simplify(tau{i}));
    fprintf("the tau of link %d is:",i);
    disp(tau{i});

    f_next=f{i};
    tau_next=tau{i};
end

% ------ generalized forces u -------
for i=1:num_joints
    u{i}=sigma{i} * (f{i}.'*(R{i}.'*[0;0;1])) + (1-sigma{i})*(tau{i}.'*(R{i}.'*[0;0;1]));
    u{i}=vpa(simplify(u{i}));
    fprintf("u of link %d is:",i);
    disp(u{i})
end
