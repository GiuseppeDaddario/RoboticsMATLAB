%Function for building the rotation matrix R, given a sequence of angles in
%a defined Euler Angles form, the order. [Ex.(ZYZ)]. If the angle is not
%given, it displays the generic rotation matrix with the defined order,
%useful for solving the inverse problem to find the angles. In this case,
%it shows also the starting point to compute the sin/cos of one angle by
%sqaring and summing the matrix components.
function [R, Rs] = getRfromEulerAngles(order, angles)

    if nargin < 2
        angles = [];
    end

    syms a1 a2 a3 real;
    theta_symbols = [a1, a2, a3];
    
    %Defining the matrices
    Rx = @(theta) [1, 0, 0;
                   0, cos(theta), -sin(theta);
                   0, sin(theta), cos(theta)];
    
    Ry = @(theta) [cos(theta), 0, sin(theta);
                   0, 1, 0;
                   -sin(theta), 0, cos(theta)];
    
    Rz = @(theta) [cos(theta), -sin(theta), 0;
                   sin(theta), cos(theta), 0;
                   0, 0, 1];
    
    %compute the matrix
    Rs = cell(1,length(order));
    R = eye(3, 'sym');
    for i = 1:length(order)
        axis = order(i);
        angle = theta_symbols(i); 
        fprintf("R{%d}=\n",i);
        switch axis
            case 'X'
                R_ = Rx(angle);
                disp(R_)
                Rs{i} = R;
                R = R * R_;
            case 'Y'
                R_ = Ry(angle);
                disp(R_)
                Rs{i} = R;
                R = R * R_;
            case 'Z'
                R_ = Rz(angle);
                disp(R_)
                Rs{i} = R;
                R = R * R_;
            otherwise
                error('Asse di rotazione non valido. Usa solo X, Y o Z.');
        end
    end
    fprintf("I compute R by multiplying the matrices:\nR=\n");
    disp(R)

    % Double the matrix if angles are given
    if ~isempty(angles)
        R = subs(R, theta_symbols(1:length(angles)), angles);
        R = double(R);
    else
        %Solve the inverse problem
        R = simplify(R);
        operations = {};

        r11 = R(1, 1);
        r12 = R(1, 2);
        r13 = R(1, 3);
        r21 = R(2, 1);
        r22 = R(2, 2);
        r23 = R(2, 3);
        r31 = R(3, 1);
        r32 = R(3, 2);
        r33 = R(3, 3);
    
        sin_cos_angles = struct();
        elements = {r11, r12, r13, r21, r22, r23, r31, r32, r33};
        index = 1;
        element_names = {'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'};

        %for each element, try to sqare and sum and see if the result is a
        %sin(angle) or cos(angle)
        for i = 1:length(elements)
            for j = i+1:length(elements)
                element1 = elements{i};
                element2 = elements{j};
                name1 = element_names{i};
                name2 = element_names{j};
               
                %compute the root
                root = sqrt(element1^2 + element2^2);
                root = simplify(root);
                
                % Check the result
                if (contains(char(root), 'abs(sin') || contains(char(root), 'abs(cos')) && ~contains(char(root),'*') && ~contains(char(root),'+')
                    if contains(char(root), 'abs(sin')
                        sin_cos_angles.(['sin_a' num2str(index)]) = root;
                        operations{end+1} = sprintf('sqrt(%s^2 + %s^2)', name1,name2);
                    elseif contains(char(root), 'abs(cos')
                        sin_cos_angles.(['cos_a' num2str(index)]) = root;
                        operations{end+1} = sprintf('sqrt(%s^2 + %s^2)', name1,name2);
                    end
                    index = index + 1;
                end
            end
        end

        disp('Sin or Cos expressions for the starting angle:');
        disp('--------------------------------------');
        
        % Visualizza le espressioni sin e cos con la relazione che le ha generate
        fnames = fieldnames(sin_cos_angles);
        for k = 1:length(operations)
            disp([char(sin_cos_angles.(fnames{k})) ' = ', operations{k}]);
        end
    end
end
