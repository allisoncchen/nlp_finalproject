function g = exponential_10_grad(x)
    assert(length(x) == 10, 'grad_P10_Exponential_10: x must be length 10');
    e = exp(x(1));
    g = zeros(10,1);
    
    % Derivative w.r.t. x(1): d/dx1 [ (e^x1 -1)/(e^x1 +1) + 0.1 e^{-x1} ]
    g(1) = (2*e) / (e + 1)^2  -  0.1 * exp(-x(1));
    
    % Derivative w.r.t. x(i) for i >= 2: 4*(x_i - 1)^3
    g(2:end) = 4 * (x(2:end) - 1).^3;
end