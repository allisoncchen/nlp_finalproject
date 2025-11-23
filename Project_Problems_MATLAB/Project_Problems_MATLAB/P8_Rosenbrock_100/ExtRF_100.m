function f = ExtRF_100(x)
    % Compute the extended Rosenbrock function: sum_{i=1}^{n-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    assert(length(x) == 100, 'P7_Rosenbrock_100: x must be of length 100 (n=100)');
    n = length(x);
    f = sum(100 * (x(2:n) - x(1:n-1).^2).^2 + (1 - x(1:n-1)).^2);
end
