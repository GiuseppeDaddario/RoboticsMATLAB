function J_pinv = pinv_w(J,W)
    disp("The weighted pseudoinverse formula: W^-1*J'*(J*W^-1*J')^-1")
    if isa(J,'sym')
        J_pinv = simplify(W^-1*J.'*(J*W^-1*J.')^-1);
    else
        J_pinv = W^-1*J.'*(J*W^-1*J.')^-1;
    end

end