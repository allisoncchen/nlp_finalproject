function H = exponential_1000_Hess(x)
    assert(length(x) == 1000, 'hess_P11_Exponential_1000: x must be length 1000');
    e = exp(x(1));
    H = zeros(1000,1000);
    
    % Second derivative w.r.t. x(1)
    H(1,1) = (2*e*(1 - e^2)) / (1 + e)^4  +  0.1*exp(-x(1));
    
    % For i >= 2: second deriv of (x_i-1)^4 is 12(x_i-1)^2
    H(2:end,2:end) = diag(12 * (x(2:end) - 1).^2);
end