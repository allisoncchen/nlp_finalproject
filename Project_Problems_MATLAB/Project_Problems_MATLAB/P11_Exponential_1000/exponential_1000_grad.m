function g = exponential_1000_grad(x)
    assert(length(x) == 1000, 'grad_P11_Exponential_1000: x must be length 1000');
    e = exp(x(1));
    g = zeros(1000,1);
    
    g(1) = (2*e) / (e + 1)^2  -  0.1 * exp(-x(1));
    g(2:end) = 4 * (x(2:end) - 1).^3;
end