function S = skew_from_vector(q)
    S = [  0   -q(3)  q(2);
          q(3)   0   -q(1);
         -q(2)  q(1)   0  ];
end