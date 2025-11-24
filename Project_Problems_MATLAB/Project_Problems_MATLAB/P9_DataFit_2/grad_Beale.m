function grad = grad_Beale(x)
    % Gradient for Beale function
    x1 = x(1);
    x2 = x(2);
    y = [1.5; 2.25; 2.625];
    powers = [1; 2; 3];
    a = 1 - x2 .^ powers;
    term = y - x1 * a;
    grad = zeros(2, 1);
    grad(1) = sum(-2 * term .* a);
    grad(2) = sum(-2 * term .* (-x1 * powers .* x2 .^ (powers - 1)));
end
