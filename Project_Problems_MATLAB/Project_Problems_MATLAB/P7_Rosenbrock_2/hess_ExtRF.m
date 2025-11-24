function H = hess_ExtRF(x)
% Hessian for extended Rosenbrock
    n = length(x);
    assert(n == 2, 'P7_Rosenbrock_2: x must be of length 2 (n=2)');
    H = zeros(n,n);
    H(1,1) = 1200*x(1)^2 - 400*x(2) + 2;
    H(1,2) = -400*x(1);
    H(2,1) = H(1,2);
    for i=2:n-1
        H(i,i) = 200 + 1200*x(i)^2 - 400*x(i+1) + 2;
        H(i,i-1) = -400*x(i-1);
        H(i-1,i) = H(i,i-1);
        H(i,i+1) = -400*x(i);
        H(i+1,i) = H(i,i+1);
    end
    if n > 1
        H(n,n) = 200;
        H(n,n-1) = -400*x(n-1);
        H(n-1,n) = H(n,n-1);
    end
end