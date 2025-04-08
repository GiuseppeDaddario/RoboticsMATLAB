function eqs = trajectory(mode,conditions,T, n_joints,customMode)
if mode == "cubicTime" || mode == "cubic" || mode == "cubicTimeCustom"
    init = 1;
    k = 4;
    eqs = cell(n_joints,3);
    for i=1:n_joints
         [position, velocity,acceleration] = cubic(mode,conditions(init:k),T,customMode);
         eqs{i,1} = position;
         eqs{i,2} = velocity;
         eqs{i,3} = acceleration;
        init = k+1;
        k = k+4;
    end
elseif mode == "quintic"
    init = 1;
    k = 6;
    eqs = cell(n_joints,3);
    for i=1:n_joints
         [position, velocity, acceleration] = quintic(conditions(init:k),T);
         eqs{i,1} = position;
         eqs{i,2} = velocity;
         eqs{i,3} = acceleration;
        init = k+1;
        k = k+6;
    end
end
end