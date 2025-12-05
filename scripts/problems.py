import numpy as np
from scipy import sparse

############## HELPER #######################################################
def sprandsym(n, density, eig_min, kind=1):
    """
    Python approximation of MATLAB sprandsym(n, density, eig_min, kind=1).

    eig_min = reciprocal of condition number = lambda_min / lambda_max.
    Generates a random symmetric sparse matrix whose eigenvalues lie in
    [eig_min, 1], approximately.

    Strategy:
        A = R @ diag(eigs) @ R^T
        eigs evenly spaced between [eig_min, 1]
        R is a random dense matrix with orthonormal columns
    """
    np.random.seed(0)
    # Generate random sparse matrix for structure
    S = sparse.random(n, n, density=density, data_rvs=np.random.randn)
    A = (S + S.T) / 2  # enforce symmetry

    # Compute a dense symmetric matrix from structure
    A = A.toarray()

    # Random orthonormal basis
    R, _ = np.linalg.qr(np.random.randn(n, n))

    # eigenvalues in desired interval
    eigs = np.linspace(eig_min, 1, n)

    # Construct symmetric matrix with controlled spectrum
    Q = R @ np.diag(eigs) @ R.T
    return Q

############# PROBLEM 1 ##########################
def quad_10_10_func(x):
    np.random.seed(0)  # emulate MATLAB rng(0)
    q = np.random.randn(10, 1)
    Q = sprandsym(10, 0.5, 0.1, 1)

    x = x.reshape(-1, 1)
    f = 0.5 * x.T @ Q @ x + q.T @ x
    return float(f)


def quad_10_10_grad(x):
    np.random.seed(0)
    q = np.random.randn(10, 1)
    Q = sprandsym(10, 0.5, 0.1, 1)

    x = x.reshape(-1, 1)
    g = Q @ x + q
    return g.flatten()


def quad_10_10_Hess(x):
    np.random.seed(0)
    q = np.random.randn(10, 1)
    Q = sprandsym(10, 0.5, 0.1, 1)
    return Q

############# PROBLEM 2 ##########################
def quad_10_1000_func(x):
    np.random.seed(0)  # emulate MATLAB rng(0)
    q = np.random.randn(10, 1)
    Q = sprandsym(10, 0.5, 1e-3, 1)

    x = x.reshape(-1, 1)
    f = 0.5 * x.T @ Q @ x + q.T @ x
    return float(f)


def quad_10_1000_grad(x):
    np.random.seed(0)
    q = np.random.randn(10, 1)
    Q = sprandsym(10, 0.5, 1e-3, 1)

    x = x.reshape(-1, 1)
    g = Q @ x + q
    return g.flatten()


def quad_10_1000_Hess(x):
    np.random.seed(0)
    q = np.random.randn(10, 1)
    Q = sprandsym(10, 0.5, 1e-3, 1)
    return Q

############# PROBLEM 3 ##########################
def quad_1000_10_func(x):
    np.random.seed(0)  # emulate MATLAB rng(0)
    q = np.random.randn(1000, 1)
    Q = sprandsym(1000, 0.5, 0.1, 1)

    x = x.reshape(-1, 1)
    f = 0.5 * x.T @ Q @ x + q.T @ x
    return float(f)


def quad_1000_10_grad(x):
    np.random.seed(0)
    q = np.random.randn(1000, 1)
    Q = sprandsym(1000, 0.5, 0.1, 1)

    x = x.reshape(-1, 1)
    g = Q @ x + q
    return g.flatten()


def quad_1000_10_Hess(x):
    np.random.seed(0)
    q = np.random.randn(1000, 1)
    Q = sprandsym(1000, 0.5, 0.1, 1)
    return Q

############# PROBLEM 4 ##########################
def quad_1000_1000_func(x):
    np.random.seed(0)  # emulate MATLAB rng(0)
    q = np.random.randn(1000, 1)
    Q = sprandsym(1000, 0.5, 1e-3, 1)

    x = x.reshape(-1, 1)
    f = 0.5 * x.T @ Q @ x + q.T @ x
    return float(f)


def quad_1000_1000_grad(x):
    np.random.seed(0)
    q = np.random.randn(1000, 1)
    Q = sprandsym(1000, 0.5, 1e-3, 1)

    x = x.reshape(-1, 1)
    g = Q @ x + q
    return g.flatten()


def quad_1000_1000_Hess(x):
    np.random.seed(0)
    q = np.random.randn(1000, 1)
    Q = sprandsym(1000, 0.5, 1e-3, 1)
    return Q

############# PROBLEM 5 ##########################
def quartic_1_func(x):
    Q = np.array([[5.0, 1.0, 0.0, 0.5],
                  [1.0, 4.0, 0.5, 0.0],
                  [0.0, 0.5, 3.0, 0.0],
                  [0.5, 0.0, 0.0, 2.0]])
    
    sigma = 1e-4
    
    x = x.reshape(-1, 1) if x.ndim == 1 else x # ensure column vector
    f = 0.5 * (x.T @ x) + sigma / 4.0 * (x.T @ Q @ x)**2
    return float(f)


def quartic_1_grad(x):
    Q = np.array([[5.0, 1.0, 0.0, 0.5],
                  [1.0, 4.0, 0.5, 0.0],
                  [0.0, 0.5, 3.0, 0.0],
                  [0.5, 0.0, 0.0, 2.0]])
    
    sigma = 1e-4
    
    x = x.reshape(-1, 1) if x.ndim == 1 else x # ensure column vector
    Qx = Q @ x
    xTQx = x.T @ Qx
    g = x + sigma * xTQx * Qx 
    return g.flatten()


def quartic_1_Hess(x):
    Q = np.array([[5.0, 1.0, 0.0, 0.5],
                  [1.0, 4.0, 0.5, 0.0],
                  [0.0, 0.5, 3.0, 0.0],
                  [0.5, 0.0, 0.0, 2.0]])
    
    sigma = 1e-4
    
    x = x.reshape(-1, 1) if x.ndim == 1 else x # ensure column vector
    Qx = Q @ x
    xTQx = float(x.T @ Qx)
    
    H = np.eye(4) + sigma * (2 * np.outer(Qx, Qx) + xTQx * Q)
    return H

############# PROBLEM 6 ##########################
def quartic_2_func(x):
    Q = np.array([[5.0, 1.0, 0.0, 0.5],
                  [1.0, 4.0, 0.5, 0.0],
                  [0.0, 0.5, 3.0, 0.0],
                  [0.5, 0.0, 0.0, 2.0]])
    
    sigma = 1e4
    
    x = x.reshape(-1, 1) if x.ndim == 1 else x # ensure column vector
    f = 0.5 * (x.T @ x) + sigma / 4.0 * (x.T @ Q @ x)**2
    return float(f)


def quartic_2_grad(x):
    Q = np.array([[5.0, 1.0, 0.0, 0.5],
                  [1.0, 4.0, 0.5, 0.0],
                  [0.0, 0.5, 3.0, 0.0],
                  [0.5, 0.0, 0.0, 2.0]])
    
    sigma = 1e4
    
    x = x.reshape(-1, 1) if x.ndim == 1 else x # ensure column vector
    Qx = Q @ x
    xTQx = x.T @ Qx
    g = x + sigma * xTQx * Qx 
    return g.flatten()


def quartic_2_Hess(x):
    Q = np.array([[5.0, 1.0, 0.0, 0.5],
                  [1.0, 4.0, 0.5, 0.0],
                  [0.0, 0.5, 3.0, 0.0],
                  [0.5, 0.0, 0.0, 2.0]])
    
    sigma = 1e4
    
    x = x.reshape(-1, 1) if x.ndim == 1 else x # ensure column vector
    Qx = Q @ x
    xTQx = float(x.T @ Qx)
    
    H = np.eye(4) + sigma * (2 * np.outer(Qx, Qx) + xTQx * Q)
    return H

############# PROBLEM 7 ##########################
def ExtRF(x_vector): 
    assert len(x_vector) == 2, 'ExtRF: x must be of length 2 (n=2)'
    i = 0 
    summation = 0 
    n = len(x_vector)

    for i in range(n - 1): 
        summation += 100 * np.square(x_vector[i + 1] - np.square(x_vector[i])) + np.square(1 - x_vector[i])

    return summation 


def grad_ExtRF(x_vector): 
    i = 0 
    length = len(x_vector)
    gradient_vector = np.zeros(length)
    n = len(x_vector)

    
    for i in range(n): 
        if i == 0: 
            gradient_vector[i] = -400 * x_vector[i] * (x_vector[i + 1] - np.square(x_vector[i])) - 2 * ( 1 - x_vector[i])
        elif i < n - 1:
            gradient_vector[i] = -400 * x_vector[i] * (x_vector[i + 1] - np.square(x_vector[i])) - 2 * ( 1 - x_vector[i]) + 200 * (x_vector[i] - np.square(x_vector[i - 1]))
        else: 
            gradient_vector[i] = 200 * (x_vector[i] - np.square(x_vector[i - 1]))
   
    return gradient_vector


def hess_ExtRF(x_vector):
    hessian_matrix = np.zeros((len(x_vector), len(x_vector))) # n by n
    last_row = len(x_vector) - 1
    
    for i in range(len(hessian_matrix)): 
        cur_val = x_vector[i]
            
        # the first row
        if i == 0: 
            next_val = x_vector[i + 1]
            hessian_matrix[i][0] = -400 * next_val + 1200 * np.square(cur_val) + 2
            hessian_matrix[i][1] = -400 * cur_val 
        
        # last row 
        elif i == last_row: 
            hessian_matrix[last_row][last_row - 1] = -400 * x_vector[last_row - 1]
            hessian_matrix[last_row][last_row] = 200
        
        # everything in between 
        else: 
            next_val = x_vector[i + 1]
            prev_val = x_vector[i - 1]
            hessian_matrix[i][i - 1] = -400 * prev_val 
            hessian_matrix[i][i] = 200 - 400 * next_val + 1200 * np.square(cur_val) + 2
            hessian_matrix[i][i + 1] = -400 * cur_val 

            hessian_matrix[i - 1][i] = hessian_matrix[i][i - 1]
            hessian_matrix[i + 1][i] = hessian_matrix[i][i + 1]
         
    # expecting a (n-1) x (n-1) matrix, where n = len(x_vector)
    return hessian_matrix 

############# PROBLEM 8 ##########################
def ExtRF_100(x_vector): 
    assert len(x_vector) == 100, 'ExtRF_100: x must be of length 100 (n=100)'
    i = 0 
    summation = 0 
    n = len(x_vector)

    for i in range(n - 1): 
        summation += 100 * np.square(x_vector[i + 1] - np.square(x_vector[i])) + np.square(1 - x_vector[i])

    return summation 


def grad_ExtRF_100(x_vector): 
    i = 0 
    length = len(x_vector)
    gradient_vector = np.zeros(length)
    n = len(x_vector)

    
    for i in range(n): 
        if i == 0: 
            gradient_vector[i] = -400 * x_vector[i] * (x_vector[i + 1] - np.square(x_vector[i])) - 2 * ( 1 - x_vector[i])
        elif i < n - 1:
            gradient_vector[i] = -400 * x_vector[i] * (x_vector[i + 1] - np.square(x_vector[i])) - 2 * ( 1 - x_vector[i]) + 200 * (x_vector[i] - np.square(x_vector[i - 1]))
        else: 
            gradient_vector[i] = 200 * (x_vector[i] - np.square(x_vector[i - 1]))
   
    return gradient_vector


def hess_ExtRF_100(x_vector):
    hessian_matrix = np.zeros((len(x_vector), len(x_vector))) # n by n
    last_row = len(x_vector) - 1
    
    for i in range(len(hessian_matrix)): 
        cur_val = x_vector[i]
            
        # the first row
        if i == 0: 
            next_val = x_vector[i + 1]
            hessian_matrix[i][0] = -400 * next_val + 1200 * np.square(cur_val) + 2
            hessian_matrix[i][1] = -400 * cur_val 
        
        # last row 
        elif i == last_row: 
            hessian_matrix[last_row][last_row - 1] = -400 * x_vector[last_row - 1]
            hessian_matrix[last_row][last_row] = 200
        
        # everything in between 
        else: 
            next_val = x_vector[i + 1]
            prev_val = x_vector[i - 1]
            hessian_matrix[i][i - 1] = -400 * prev_val 
            hessian_matrix[i][i] = 200 - 400 * next_val + 1200 * np.square(cur_val) + 2
            hessian_matrix[i][i + 1] = -400 * cur_val 

            hessian_matrix[i - 1][i] = hessian_matrix[i][i - 1]
            hessian_matrix[i + 1][i] = hessian_matrix[i][i + 1]
         
    # expecting a (n-1) x (n-1) matrix, where n = len(x_vector)
    return hessian_matrix 


############# PROBLEM 9 ##########################
def Beale_function(x_vector): 
    summation = 0 

    y_vector = [1.5, 2.25, 2.625] 
    for i in range(3): 
        power = i + 1
        summation += np.square(y_vector[i] - (x_vector[0] * (1 - x_vector[1] ** power)))

    return summation

def Beale_gradient(x_vector): 
    i = 0 
    gradient_vector = np.zeros(2)
    y_vector = [1.5, 2.25, 2.625]

    for i in range(3): 
        power = i + 1
        gradient_vector[0] += 2 * (y_vector[i] - (x_vector[0] * (1 - x_vector[1] ** power))) * (-1) * (1 - x_vector[1] ** power)
        gradient_vector[1] += 2 * (y_vector[i] - (x_vector[0] * (1 - x_vector[1] ** power))) * (-x_vector[0]) * power * (-x_vector[1] ** (power - 1)) 
        
    return gradient_vector

def Beale_hessian(x_vector): 
    hessian_matrix = np.zeros((2, 2))
    x1, x2 = x_vector
    y_vector = [1.5, 2.25, 2.625]

    for i in range(3):
        p = i + 1
        residual = y_vector[i] - x1 * (1 - x2**p)

        # (0,0)
        hessian_matrix[0,0] += 2 * (1 - x2**p)**2

        # (0,1) and (1,0)
        hessian_matrix[0,1] += 2 * ((-1) * (1 - x2 ** p) * (x1 * p * x2 ** (p - 1)) + residual * (p * x2 ** (p - 1)))
        hessian_matrix[1,0] = hessian_matrix[0,1]

        # (1,1)
        partial = x1 * p * (p - 1) * x2 ** (p - 2) if p > 1 else 0
        hessian_matrix[1,1] += 2 * ((x1 * p * x2 ** (p - 1)) ** 2 + residual * partial)

    return hessian_matrix

############# PROBLEM 10 ##########################
def exponential_10_func(x):
    x = np.asarray(x).flatten()
    assert len(x) == 10, 'exponential_10_func: x must be length 10'
    
    e = np.exp(x[0])
    term1 = (e - 1) / (e + 1)
    term2 = 0.1 * np.exp(-x[0])
    term3 = np.sum((x[1:] - 1)**4)
    
    return float(term1 + term2 + term3)


def exponential_10_grad(x):
    x = np.asarray(x).flatten()
    assert len(x) == 10, 'exponential_10_grad: x must be length 10'
    
    e = np.exp(x[0])
    g = np.zeros(10)
    
    # d/dx1 of first two terms
    g[0] = (2 * e) / (e + 1)**2 - 0.1 * np.exp(-x[0])
    
    # d/dxi for i >= 2: 4*(xi - 1)^3
    g[1:] = 4 * (x[1:] - 1)**3
    
    return g


def exponential_10_Hess(x):
    x = np.asarray(x).flatten()
    assert len(x) == 10, 'exponential_10_Hess: x must be length 10'
    
    e = np.exp(x[0])
    H = np.zeros((10, 10))
    
    H[0,0] = 2 * e * (1 - e) / (e + 1)**3 + 0.1 * np.exp(-x[0])
    
    # For i >= 2: second derivative of (xi - 1)^4 is 12*(xi - 1)^2
    # So diagonal entries only
    diag_values = 12 * (x[1:] - 1)**2
    np.fill_diagonal(H[1:, 1:], diag_values)
    
    return H

############# PROBLEM 11 ##########################
def exponential_1000_func(x):
    x = np.asarray(x).flatten()
    assert len(x) == 1000, 'exponential_1000_func: x must be length 1000'
    
    e = np.exp(x[0])
    term1 = (e - 1) / (e + 1)
    term2 = 0.1 * np.exp(-x[0])
    term3 = np.sum((x[1:] - 1)**4)
    
    return float(term1 + term2 + term3)


def exponential_1000_grad(x):
    x = np.asarray(x).flatten()
    assert len(x) == 1000, 'exponential_1000_grad: x must be length 1000'
    
    e = np.exp(x[0])
    g = np.zeros(1000)
    
    # d/dx1 of first two terms
    g[0] = (2 * e) / (e + 1)**2 - 0.1 * np.exp(-x[0])
    
    # d/dxi for i >= 2: 4*(xi - 1)^3
    g[1:] = 4 * (x[1:] - 1)**3
    
    return g


def exponential_1000_Hess(x):
    x = np.asarray(x).flatten()
    assert len(x) == 1000, 'exponential_1000_Hess: x must be length 1000'
    
    e = np.exp(x[0])
    H = np.zeros((1000, 1000))
    
    H[0,0] = 2 * e * (1 - e) / (e + 1)**3 + 0.1 * np.exp(-x[0])
    
    # For i >= 2: second derivative of (xi - 1)^4 is 12*(xi - 1)^2
    # So diagonal entries only
    diag_values = 12 * (x[1:] - 1)**2
    np.fill_diagonal(H[1:, 1:], diag_values)
    
    return H

############# PROBLEM 12 ##########################
def genhumps_5_func(x):
    x = np.asarray(x).flatten()
    assert len(x) == 5, 'genhumps_5_func: x must be of length 5'
    
    f = 0.0
    for i in range(4):  # i from 0 to 3 in Python
        xi = x[i]
        xi1 = x[i+1]
        f += np.sin(2*xi)**2 * np.sin(2*xi1)**2 + 0.05 * (xi**2 + xi1**2)
    
    return float(f)


def genhumps_5_grad(x):
    x = np.asarray(x).flatten()
    assert len(x) == 5, 'genhumps_5_grad: x must be of length 5'
    
    s1 = np.sin(2*x[0])
    c1 = np.cos(2*x[0])
    s2 = np.sin(2*x[1])
    c2 = np.cos(2*x[1])
    s3 = np.sin(2*x[2])
    c3 = np.cos(2*x[2])
    s4 = np.sin(2*x[3])
    c4 = np.cos(2*x[3])
    s5 = np.sin(2*x[4])
    c5 = np.cos(2*x[4])
    
    g = np.zeros(5)
    
    g[0] = 4 * s1 * c1 * s2**2 + 0.1 * x[0]
    g[1] = 4 * s2 * c2 * (s1**2 + s3**2) + 0.2 * x[1]
    g[2] = 4 * s3 * c3 * (s2**2 + s4**2) + 0.2 * x[2]
    g[3] = 4 * s4 * c4 * (s3**2 + s5**2) + 0.2 * x[3]
    g[4] = 4 * s5 * c5 * s4**2 + 0.1 * x[4]
    
    return g


def genhumps_5_Hess(x):
    x = np.asarray(x).flatten()
    assert len(x) == 5, 'genhumps_5_Hess: x must be of length 5'
    
    s1 = np.sin(2*x[0])
    c1 = np.cos(2*x[0])
    s2 = np.sin(2*x[1])
    c2 = np.cos(2*x[1])
    s3 = np.sin(2*x[2])
    c3 = np.cos(2*x[2])
    s4 = np.sin(2*x[3])
    c4 = np.cos(2*x[3])
    s5 = np.sin(2*x[4])
    c5 = np.cos(2*x[4])
    
    H = np.zeros((5, 5))
    
    # Diagonal terms
    H[0,0] = 8 * s2**2 * (c1**2 - s1**2) + 0.1
    H[1,1] = 8 * (s1**2 + s3**2) * (c2**2 - s2**2) + 0.2
    H[2,2] = 8 * (s2**2 + s4**2) * (c3**2 - s3**2) + 0.2
    H[3,3] = 8 * (s3**2 + s5**2) * (c4**2 - s4**2) + 0.2
    H[4,4] = 8 * s4**2 * (c5**2 - s5**2) + 0.1
    
    # Off-diagonal terms (upper triangle)
    H[0,1] = 16 * s1 * c1 * s2 * c2
    H[1,2] = 16 * s2 * c2 * s3 * c3
    H[2,3] = 16 * s3 * c3 * s4 * c4
    H[3,4] = 16 * s4 * c4 * s5 * c5
    
    # Fill lower triangle (symmetric)
    H[1,0] = H[0,1]
    H[2,1] = H[1,2]
    H[3,2] = H[2,3]
    H[4,3] = H[3,4]
    
    return H