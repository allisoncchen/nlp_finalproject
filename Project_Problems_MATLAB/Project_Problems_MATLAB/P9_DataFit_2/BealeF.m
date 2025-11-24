function f = BealeF(x)
    % Compute the Beale function
    x1 = x(1);
    x2 = x(2);
    y = [1.5; 2.25; 2.625];
    powers = [1; 2; 3];
    a = 1 - x2 .^ powers;
    f = sum((y - x1 * a).^2);
end

