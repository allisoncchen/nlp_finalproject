% Code written by: Nabla Ninjas 

% Problem Number: 6
% Problem Name: quartic_2
% Problem Description: A quartic function. Dimension n = 4

% function that computes the Hessian of the quartic_2 function
function [H] = quartic_2_Hess(x)

% Matrix Q
Q = [5 1 0 0.5;
     1 4 0.5 0;
     0 0.5 3 0;
     0.5 0 0 2];
 
% Set sigma value
sigma = 1e4;

% compute Hess = I + sigma * [2*(Qx)*(Qx)' + (x'*Q*x)*Q]
H = eye(4) + sigma * (2 * ((Q * x) * (Q * x)') + (x' * Q * x) * Q);

end
