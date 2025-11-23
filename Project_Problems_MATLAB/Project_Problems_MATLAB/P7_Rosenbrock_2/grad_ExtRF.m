function grad = grad_ExtRF(x)
    % Gradient for extended Rosenbrock
    n = length(x);
    assert(n == 2, 'P7_Rosenbrock_2: x must be of length 2 (n=2)');
    grad = zeros(n, 1);
    grad(1) = 400 * x(1) * (x(1)^2 - x(2)) + 2 * (x(1) - 1); % d/dx_1
    for i = 2:n-1
        grad(i) = -200 * (x(i-1)^2 - x(i)) + 400 * x(i) * (x(i)^2 - x(i+1)) + 2 * (x(i) - 1); % d/dx_i
    end
    if n > 1
        grad(n) = -200 * (x(n-1)^2 - x(n)); % d/dx_n
    end
end

