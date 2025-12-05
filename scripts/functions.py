import math 
import numpy as np 
import sys


# Calculating the Rosenbrock function given a x vector 
# Output: scalar
def Rosenbrock_function(x_vector): 
    i = 0 
    summation = 0 
    n = len(x_vector)

    for i in range(n - 1): 
        summation += 100 * np.square(x_vector[i + 1] - np.square(x_vector[i])) + np.square(1 - x_vector[i])

    return summation 


# Calculating the Rosenbrock gradient
# Output: vector 
def Rosenbrock_gradient(x_vector): 
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


# Calculating the Rosenbrock hessian 
# Output: matrix 
def Rosenbrock_hessian(x_vector):
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


# Calculating the Rosenbrock function given a x vector 
# Output: scalar
def Beale_function(x_vector): 
    summation = 0 

    y_vector = [1.5, 2.25, 2.625] 
    for i in range(3): 
        power = i + 1
        summation += np.square(y_vector[i] - (x_vector[0] * (1 - x_vector[1] ** power)))

    return summation


# Calculating the Rosenbrock gradient given a x vector 
# Output: vector
def Beale_gradient(x_vector): 
    i = 0 
    gradient_vector = np.zeros(2)
    y_vector = [1.5, 2.25, 2.625]

    for i in range(3): 
        power = i + 1
        gradient_vector[0] += 2 * (y_vector[i] - (x_vector[0] * (1 - x_vector[1] ** power))) * (-1) * (1 - x_vector[1] ** power)
        gradient_vector[1] += 2 * (y_vector[i] - (x_vector[0] * (1 - x_vector[1] ** power))) * (-x_vector[0]) * power * (-x_vector[1] ** (power - 1)) 
        
    return gradient_vector


# Calculating the Rosenbrock hessian given a x vector 
# Output: matrix 
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



# Calculating the unnamed function given a x vector 
# Output: scalar
def unnamed_function(x_vector): 
    n = len(x_vector)
    summation = np.square(x_vector[0])

    for i in range(n - 1): 
        summation += (x_vector[i] - x_vector[i + 1]) ** (2 * (i + 1))
        
    
    return summation


# Calculating the function given a x vector 
# Output: vector
def unnamed_gradient(x_vector): 
    n = len(x_vector)
    gradient_vector = np.zeros(n)

    for i in range(n): 
        if i == 0: 
            gradient_vector[i] = 2 * x_vector[i] + 2 * (x_vector[i] - x_vector[i + 1])
        elif i == (n - 1): 
            gradient_vector[i] = (i * 2) * (-1) * ((x_vector[i - 1] - x_vector[i]) ** (i * 2 - 1))
        else: 
            gradient_vector[i] = (i * 2) * (-1) * (x_vector[i - 1] - x_vector[i]) ** (i * 2 - 1) + ((i + 1) * 2) * (x_vector[i] - x_vector[i + 1]) ** (((i + 1) * 2) - 1) 


    return gradient_vector



# Calculating the Rosenbrock function given a x vector 
# Output: matrix
def unnamed_hessian(x_vector): 
    n = len(x_vector)
    hessian_matrix = np.zeros(((n, n)))


    for i in range(n):  

        cur_val = x_vector[i]

        if i == 0: 
            hessian_matrix[i][0] = 4
            hessian_matrix[i][1] = -2 
        
        elif i == (n - 1): 
            prev_val = x_vector[i - 1]
            hessian_matrix[i][i-1] = (i * 2) * (-1) * (i * 2 - 1) * (prev_val - cur_val ) ** (2 * i - 2)
            hessian_matrix[i][i] = (i * 2) * (i * 2 - 1) * (prev_val - cur_val ) ** (i * 2 - 2)

        else: 
            next_val = x_vector[i + 1]
            prev_val = x_vector[i - 1]
            coef = i * 2 
            power = coef - 1
            base_left = (prev_val - cur_val)
            base_right = (cur_val - next_val)
            coef_right = (i + 1) * 2
            power_right = (coef_right) - 1

        
            hessian_matrix[i][i - 1] = (-1) * (coef * power) * base_left ** (power - 1)
            hessian_matrix[i][i] = ((coef * power) * base_left ** (power - 1)) + ((coef_right * power_right) * (base_right) ** (power_right - 1))
            hessian_matrix[i][i + 1] = (-1) * (coef_right * power_right) * base_right ** (power_right - 1)

    
    return hessian_matrix
