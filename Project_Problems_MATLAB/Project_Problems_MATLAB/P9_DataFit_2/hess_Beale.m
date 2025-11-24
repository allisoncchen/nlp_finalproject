function H = hess_Beale(x)
    % HESS_BEALE  Hessian of Beale function
    %   x is 2x1 vector [x1; x2]
    %   f(x) = sum_{i=1}^3 ( y_i - x1*(1 - x2^i) )^2, y = [1.5;2.25;2.625]
    
    x1 = x(1);
    x2 = x(2);
    y = [1.5; 2.25; 2.625];
    powers = (1:3).'; % column vector [1;2;3]
    
    a = 1 - x2.^powers; % a_i = 1 - x2^i
    t = y - x1 .* a; % residuals t_i
    
    b = x1 .* powers .* (x2.^(powers-1)); % when x2=0 and powers-1<0, matlab yields Inf, handle below
    
    H11 = 2 * sum( a.^2 );
    
    term_ix = powers .* (x2.^(powers-1)); % may produce Inf for x2==0 & power==1- but will be multiplied / used safely below
    
    for k = 1:length(powers)
        if ~isfinite(term_ix(k))
            term_ix(k) = 0;
        end
    end
    
    H12 = 2 * sum( -a .* b + t .* term_ix );
    
    H22 = 0;
    for k = 1:length(powers)
        i = powers(k);
        % safe compute b(k)^2
        b2 = b(k)^2;
        if i >= 2
            % compute t * (x1 * i * (i-1) * x2^(i-2))
            extra = t(k) * ( x1 * i * (i-1) * x2^(i-2) );
        else
            extra = 0;
        end
        H22 = H22 + (b2 + extra);
    end
    H22 = 2 * H22;
    
    H = [H11, H12; H12, H22];
end
