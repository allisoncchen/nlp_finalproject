function f = exponential_1000_func(x)
    assert(length(x) == 1000, 'P10_Exponential_1000: x must be length 1000');
    term1 = (exp(x(1)) - 1) / (exp(x(1)) + 1);
    term2 = 0.1 * exp(-x(1));
    term3 = sum((x(2:end) - 1).^4);
    f = term1 + term2 + term3;
end