function dq0 = H_func(varargin)
if varargin{1} == "obs"
    dq0 = dq0_obs(varargin{2:end});
end
end

function dq0 = dq0_obs(varargin)
a  =  varargin{1}; 
Ja =  varargin{2}; 
b  =  varargin{3};

%a is the closest point of the robot
%b is the closest point of the obstacle

H = norm(a-b)^2;
nabla_H =(1/norm(a-b) * Ja.' * (a-b));
dq0 = simplify(nabla_H);

end