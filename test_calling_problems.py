import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

# Add folder to MATLAB path
eng.addpath(eng.genpath('/home/parul/nlp/nabla_ninjas/nlp_finalproject/Project_Problems_MATLAB/Project_Problems_MATLAB'), nargout=0)


############ Randomly testing a given MATLAB function to see if it works 
x = matlab.double([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

f = eng.quad_10_10_func(x)

print(f)

############ Checking to make sure my quartic function grad and hess are right

# test array
x = np.array([1.0, 2.0, 3.0, -1.0])
x_mat = matlab.double(x.reshape(-1,1).tolist())

# calc grad and hess 
grad_mat = np.array(eng.quartic_1_grad(x_mat)).flatten()
hess_mat = np.array(eng.quartic_1_Hess(x_mat))

print("Gradient from MATLAB:", grad_mat)
print("Hessian from MATLAB:\n", hess_mat)


# numerical gradient to compare my funcs against 
def f_python(x_numpy):
    """Call MATLAB quartic_1_func from Python for numeric testing."""
    x_mat_local = matlab.double(x_numpy.reshape(-1,1).tolist())
    return float(eng.quartic_1_func(x_mat_local))

eps = 1e-6
n = len(x)
grad_num = np.zeros(n)

for i in range(n):
    e = np.zeros(n)
    e[i] = 1.0
    grad_num[i] = (f_python(x + eps*e) - f_python(x - eps*e)) / (2*eps)

print("\nNumerical gradient:", grad_num)
print("Gradient error: ", np.linalg.norm(grad_num - grad_mat))


# numerical hessian to compare mine against 
def grad_python(x_numpy):
    x_mat_local = matlab.double(x_numpy.reshape(-1,1).tolist())
    return np.array(eng.quartic_1_grad(x_mat_local)).flatten()

H_num = np.zeros((n,n))

for i in range(n):
    e = np.zeros(n)
    e[i] = 1.0
    H_num[:, i] = (grad_python(x + eps*e) - grad_python(x - eps*e)) / (2*eps)

print("\nNumerical Hessian: ", H_num)
print("Hessian error: ", np.linalg.norm(H_num - hess_mat))

#################### Checking the exponential functions grads and hessians 

def numerical_grad(f, x, h=1e-6):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h
        g[i] = (f(x + dx) - f(x - dx)) / (2 * h)
    return g


def numerical_hess(f, x, h=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx_i = np.zeros(n); dx_i[i] = h
            dx_j = np.zeros(n); dx_j[j] = h
            H[i,j] = (
                f(x + dx_i + dx_j)
                - f(x + dx_i - dx_j)
                - f(x - dx_i + dx_j)
                + f(x - dx_i - dx_j)
            ) / (4 * h * h)
    return H

# Test point
x = np.random.randn(10)
x_mat = matlab.double(x.reshape(-1,1).tolist())

# MATLAB function handles
f_mat = lambda v: eng.exponential_10_func(matlab.double(v.reshape(-1,1).tolist()))

# Call MATLAB gradient & Hessian
g_mat = np.array(eng.exponential_10_grad(x_mat)).flatten()
H_mat = np.array(eng.exponential_10_Hess(x_mat))

# Numerical checks
g_num = numerical_grad(lambda z: f_mat(z), x)
H_num = numerical_hess(lambda z: f_mat(z), x)

print("Gradient error:", np.linalg.norm(g_mat - g_num))
print("Hessian error:", np.linalg.norm(H_mat - H_num))
