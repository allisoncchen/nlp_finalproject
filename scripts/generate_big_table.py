import math 
import numpy as np 
import sys
import time 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import scripts.problems as pr

K_MAX = 2000
TAO = 0.5
BETA = 0.0004
EMIN = 10e-8

START_ALPHA = 1
C1 = 0.0004
C2 = 0.9
TAO = 0.5
K_MAX = 1000

# ============================================================================
# Runs all problems and algorithm combinations & creates 1 table 
# ============================================================================

# Armijo condition – find optimal steplength (alpha) selection
# Modified to accept c1 as parameter
def Armijo_backtracking_param(x, func, gradient, p, alpha_init, c1): 
    i = 0 
    a = alpha_init 
    
    while i < K_MAX: 
        new_val = func(x + (a * p))
        modeled_val = func(x) + (c1 * a * float(np.dot(gradient(x).T, p)))

        # termination condition 
        if (new_val <= modeled_val): 
            return a, i + 1  # Return alpha and number of evaluations

        # Increment alpha 
        else: 
            a = TAO * a # a is getting smaller 
        
        i += 1
    
    return -100, i


# Runs Wolfe linesearch to identify the best alpha using Armijo and the curvature condition 
# Modified to accept c1 and c2 as parameters
def Wolfe_linesearch_param(x, p, func, gradient, c1, c2): 

    al = 0 
    au = float('inf') 
    a = 1
    i = 0

    while i < K_MAX:
        new_val = func(x + (a * p))
        modeled_val = func(x) + (c1 * a * float(np.dot(gradient(x).T, p)))
        
        if (new_val > modeled_val): # armijo's
            au = a
        else: 
            if (np.dot((gradient(x + (a * p)).T), p) < c2 * np.dot(gradient(x).T, p)): # curvature condition 
                al = a
            else: 
                return a, i + 1 # stopping and returning alpha as output of linesearch 
        
        if au < float('inf'): 
            a = (al + au) / 2
        else: 
            a = 2 * a
        
        i += 1
    
    return a, i


# ============================================================================
# PARAMETERIZED OPTIMIZATION METHODS
# ============================================================================

# Runs the steepest descent algorithm with Armijo backtracking 
# Modified to accept c1, c2, start_alpha as parameters
def steepest_descent_param(x, func, gradient, armijo, c1, c2, start_alpha): 
    
    alpha = start_alpha
    
    prev_vector = x
    cur_vector = x
    k = 0 # outer loop 
    stopping_point = (10**-8) * max(1, np.linalg.norm(gradient(x)))
    total_func_evals = 0

    # Stopping Condition: hit max iterations without convergence
    while (k < K_MAX): 

        norm_cur_gradient_vector = np.linalg.norm(gradient(cur_vector))

        # Stopping Condition: satisfiable convergence
        if norm_cur_gradient_vector <= stopping_point: 
            return func(cur_vector), k, norm_cur_gradient_vector, True, total_func_evals
        
        # for every iteration after the initial, recalculate alpha 
        if k > 0: 
            theta = np.dot(gradient(prev_vector), (-1 * gradient(prev_vector).T))
            alpha = 2 * (func(cur_vector) - func(prev_vector)) / theta

        # choose a descent direction p  
        p = -gradient(cur_vector)

        # find alpha
        if armijo == True: 
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, alpha, c1)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, c1, c2)

        total_func_evals += evals

        if alpha == -100: 
            return func(cur_vector), k, norm_cur_gradient_vector, False, total_func_evals
        
        # Update the the iterate 
        prev_vector = cur_vector 
        cur_vector = cur_vector + (alpha * p)

        # Set k <- k + 1
        k += 1 
    
    return func(cur_vector), k, norm_cur_gradient_vector, False, total_func_evals


# Runs Newton's method with Armijo backtracking to find alpha
# Modified to accept c1, c2, start_alpha as parameters
def newtons_method_param(x, func, gradient, hessian, armijo, c1, c2, start_alpha):
    
    cur_vector = x
    k = 0 # for the outer loop 
    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector
    total_func_evals = 0

    # Stopping Condition – hit max iterations without convergence
    while (k < K_MAX): 

        alpha = start_alpha
        cur_gradient = np.linalg.norm(gradient(cur_vector))

        # Stopping Condition – satisfiable convergence
        if cur_gradient <= stopping_point: 
            return func(cur_vector), k, cur_gradient, True, total_func_evals
    
        # Choose a descent direction p  
        gradient_vector = gradient(cur_vector)
        hessian_matrix = hessian(cur_vector)
        try: 
            p = np.linalg.solve(hessian_matrix, -gradient_vector)
        except np.linalg.LinAlgError:
            p = -gradient(cur_vector) # default to steepest descent 
            
        # Find alpha 
        if armijo == True: 
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, 1, c1)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, c1, c2)
        
        total_func_evals += evals

        if alpha == -100: 
            return func(cur_vector), k, cur_gradient, False, total_func_evals
        
        # Update the the iterate 
        cur_vector = cur_vector + (alpha * p)

        # Set k <- k + 1
        k += 1 
    
    return func(cur_vector), k, cur_gradient, False, total_func_evals


# Runs the cholesky algorithm to find the optimal addition to the hessian matrix
# Output: delta
def cholesky_with_added_multiple_of_identity(A):
    min_diagonal = 10000
    delta = 0 

    length = len(A)
    I = np.identity(length)
    t = 0 

    for i in range(len(A)): # getting the minimum diagonal of A 
        min_diagonal = min(min_diagonal, A[i][i])
    if min_diagonal > 0: 
        delta = 0 
    else: 
        delta = -min_diagonal + BETA 
    
    # apply cholesky algorithm 
    while t < K_MAX: 

        try: 
            np.linalg.cholesky(A + I * delta)
            return delta
        
        except np.linalg.LinAlgError: 

            delta = max(2*delta, BETA)

        t += 1
    
    return -100


# Runs modified Newton's method with Armijo backtracking to get the alpha and Cholesky's to get descent direction 
# Modified to accept c1, c2, start_alpha as parameters
def modified_newtons_method_param(x, func, gradient, hessian, armijo, c1, c2, start_alpha): 

    cur_vector = x
    k = 0 # outer loop 
    delta = 0
    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    total_func_evals = 0
    
    while (k < K_MAX): 
        alpha = start_alpha
        cur_gradient = np.linalg.norm(gradient(cur_vector))
        
        if cur_gradient <= stopping_point: 
            return func(cur_vector), k, cur_gradient, True, total_func_evals
    
        # delta = 0
        # Choosing a descent direction p  
        gradient_vector = gradient(cur_vector)
        hessian_matrix = hessian(cur_vector)
    
        try: 
            E = np.zeros((len(x), len(x)))
            np.linalg.cholesky(hessian_matrix)
            delta = 0 
        except np.linalg.LinAlgError: # hessian is not positive definite
            delta = cholesky_with_added_multiple_of_identity(hessian_matrix)
        
        E = np.identity(len(gradient_vector)) * delta
        # print(E)
        B = hessian_matrix + E
        p = np.linalg.solve(B, -gradient_vector)
        
        # Find alpha 
        if armijo == True: 
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, 1, c1) # Choose using Armijo backtracking line search 
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, c1, c2) # Choose using Wolfe line search 
        
        total_func_evals += evals

        if alpha == -100: 
            return func(cur_vector), k, cur_gradient, False, total_func_evals
        
        # Update the iterate 
        cur_vector = cur_vector + (alpha * p)

        # Set k <- k + 1
        k += 1 
    
    return func(cur_vector), k, cur_gradient, False, total_func_evals


# Modified BFGS
def BFGS_param(x, func, gradient, armijo, c1, c2, start_alpha): 
    length = len(x)
    H = np.identity(length)
    prev_vector = x
    cur_vector = x
    k = 0 
    I = np.identity(length)
    alpha = start_alpha

    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    total_func_evals = 0
    
    while (k < K_MAX): 
        gradient_vector = gradient(cur_vector)
        cur_gradient = np.linalg.norm(gradient(cur_vector))

        # stopping Condition, satisfiable convergence
        if cur_gradient <= stopping_point: 
            return func(cur_vector), k, cur_gradient, True, total_func_evals
        
        p = -H @ gradient_vector
        
        # choose a steplength with wolfe's
        if armijo == True: 
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, 1, c1)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, c1, c2)

        total_func_evals += evals

        if alpha == -100:
            return func(cur_vector), k, cur_gradient, False, total_func_evals

        # update the iterate 
        cur_vector = prev_vector + (alpha * p)

        # define s & y 
        s = cur_vector - prev_vector
        y = gradient(cur_vector) - gradient(prev_vector) 

        if np.dot(s, y) > (EMIN * np.linalg.norm(y) * np.linalg.norm(s)): 
            pk = 1 / (s @ y) # I don't know if this is better
            H = (I - (pk * np.outer(s, y))) @ H @ (I - pk * np.outer(y, s)) + (pk * np.outer(s, s))

        # Set k <- k + 1
        k += 1

        prev_vector = cur_vector
    
    return func(cur_vector), k, cur_gradient, False, total_func_evals


# Runs DFP with Armijo backtracking line search 
# Modified to accept c1, c2, start_alpha as parameters
def DFP_param(x, func, gradient, armijo, c1, c2, start_alpha): 
    length = len(x)
    H = np.identity(length)
    prev_vector = x
    cur_vector = x
    k = 0 
    I = np.identity(length)
    alpha = start_alpha

    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    total_func_evals = 0
    
    while (k < K_MAX): 
        gradient_vector = gradient(cur_vector)
        cur_gradient = np.linalg.norm(gradient(cur_vector))

        # stopping Condition, satisfiable convergence
        if cur_gradient <= stopping_point: 
            return func(cur_vector), k, cur_gradient, True, total_func_evals
        
        p = -H @ gradient_vector
        
        # choose a steplength with Armijo backtracking linesearch
        if (armijo): 
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, 1, c1)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, c1, c2)

        total_func_evals += evals

        if alpha == -100:
            return func(cur_vector), k, cur_gradient, False, total_func_evals

        # update the iterate 
        cur_vector = prev_vector + (alpha * p)

        # define s & y 
        s = cur_vector - prev_vector
        y = gradient(cur_vector) - gradient(prev_vector) 

        if np.dot(s, y) > (EMIN * np.linalg.norm(y) * np.linalg.norm(s)):
            sy = np.dot(s, y)
            Hy = H @ y
            yHy = np.dot(y, Hy)

            H = H + np.outer(s, s) / sy - np.outer(Hy, Hy) / yHy

        # Set k <- k + 1
        k += 1

        prev_vector = cur_vector
        
    return func(cur_vector), k, cur_gradient, False, total_func_evals


# Input is gradient at f(x_k) and last y_k, s_k pairs, replaces H_k nabla f update 
def two_loop_recursion(cur_gradient, y_s_pairs, gamma, I): 
    q = cur_gradient.copy()
    k = len(y_s_pairs)
    alphas = np.zeros(k)

    for i in range(k - 1, -1, -1): 
        pair = y_s_pairs[i]
        y = pair[0]
        s = pair[1]
        p = 1 / np.dot(y, s)

        alpha = p * np.dot(s, q) 
        alphas[i] = alpha
        q = q - (alpha * y) 

    H_0 = (gamma * I)
    r = H_0 @ q.T 
    for i in range(k): 
        pair = y_s_pairs[i]
        y = pair[0]
        s = pair[1]
        p = 1 / np.dot(y, s)

        beta = p * np.dot(y, r) 
        r = r + s * (alphas[i] - beta) 
    
    return r # this is H * gradient f


# Modified L-BFGS
def L_BFGS_param(x, func, gradient, armijo, c1, c2, start_alpha): 
    gamma = 1 # starts off as one from the specs 
    n = len(x)
    m = min(n, 10) 
    I = np.identity(n)
    alpha = start_alpha
    k = 0 
    cur_vector = x.copy()
    prev_vector = x.copy()
    

    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    y_s_pairs = [] 
    total_func_evals = 0

    while (k < K_MAX): 
        cur_gradient = gradient(cur_vector)
        cur_gradient_norm = np.linalg.norm(cur_gradient)

        # stopping Condition, satisfiable convergence
        if cur_gradient_norm <= stopping_point: 
            return func(cur_vector), k, cur_gradient_norm, True, total_func_evals

        p = -(two_loop_recursion(cur_gradient, y_s_pairs, gamma, I)) # y_s_pairs is going to be empty on first iteration
        

        # choose a steplength with wolfe's
        if armijo == True: 
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, 1, c1)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, c1, c2)

        total_func_evals += evals

        if alpha == -100:
            return func(cur_vector), k, cur_gradient_norm, False, total_func_evals

        # update the iterate 
        cur_vector = cur_vector + (alpha * p)

        # define s & y 
        s = cur_vector - prev_vector
        y = gradient(cur_vector) - cur_gradient

        if np.dot(s, y) > (EMIN * np.linalg.norm(y) * np.linalg.norm(s)): 
            gamma = np.dot(s, y) / np.dot(y, y)

            if len(y_s_pairs) >= m: # array has hit capacity 
                y_s_pairs.pop(0) # remove value at first index
            
            y_s_pairs.append([y, s])

        # Set k <- k + 1
        k += 1
        prev_vector = cur_vector
    
    return func(cur_vector), k, cur_gradient_norm, False, total_func_evals


# Runs Newton CG method with Wolfe line search 
# Modified to accept c1, c2, start_alpha as parameters
def Newton_CG_param(x, func, gradient_func, hessian, armijo, c1, c2, start_alpha): 
    k = 0 
    alpha_k = start_alpha
    cur_vector = x
    n = 0.01
    total_func_evals = 0
    
    while k < K_MAX: 
        z = 0 
        r = gradient_func(cur_vector)
        d = -r

        cur_gradient = np.linalg.norm(gradient_func(cur_vector))
        p = 0.0
        stopping_point = 1e-8 * max(1, np.linalg.norm(cur_gradient))
    
        if cur_gradient <= stopping_point: 
            return func(cur_vector), k, cur_gradient, True, total_func_evals
        
        j = 0 
        while j < K_MAX: 
            if (d.T @ hessian(cur_vector) @ d) <= 0: 
                if j == 0: 
                    p = -gradient_func(cur_vector)
                    break 
                else: 
                    p = z
                    break 
            else: 
                alpha_j = (np.dot(r, r)) / (d.T @ hessian(cur_vector) @ d)
                z = z + (alpha_j * d)
                old_r = r 
                r = r + (alpha_j * hessian(cur_vector) @ d) 
            
                if np.linalg.norm(r) <= n * np.linalg.norm(gradient_func(cur_vector)):
                    p = z
                    break

                beta = np.dot(r, r) / np.dot(old_r, old_r)
                d = -r + (beta * d)

            j += 1
        

        if armijo: 
            alpha_k, evals = Armijo_backtracking_param(cur_vector, func, gradient_func, p, 1, c1)
        else: 
            alpha_k, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient_func, c1, c2)

        total_func_evals += evals

        if alpha_k == -100:
            return func(cur_vector), k, cur_gradient, False, total_func_evals

        cur_vector = cur_vector + (alpha_k * p)
        k += 1

    return func(cur_vector), k, cur_gradient, False, total_func_evals



def compare_algorithms(): 
    problems = [
        {
            "name": "P1_quad_10_10",
            "n": 10,
            "x0": 20 * np.random.rand(10) - 10,
            "func": pr.quad_10_10_func,
            "grad": pr.quad_10_10_grad,
            "hess": pr.quad_10_10_Hess,
        },
        {
            "name": "P2_quad_10_1000",
            "n": 10,
            "x0": 20 * np.random.rand(10) - 10,
            "func": pr.quad_10_1000_func,
            "grad": pr.quad_10_1000_grad,
            "hess": pr.quad_10_1000_Hess,
        },
        {
            "name": "P3_quad_1000_10", # I think this is causing problems 
            "n": 1000,
            "x0": 20 * np.random.rand(1000) - 10,
            "func": pr.quad_1000_10_func,
            "grad": pr.quad_1000_10_grad,
            "hess": pr.quad_1000_10_Hess,
        },
        {
            "name": "P4_quad_1000_1000",
            "n": 1000,
            "x0": 20 * np.random.rand(1000) - 10,
            "func": pr.quad_1000_1000_func,
            "grad": pr.quad_1000_1000_grad,
            "hess": pr.quad_1000_1000_Hess,
        },
        {
            "name": "P5_quartic_1",
            "n": 4,
            "x0": np.array([np.cos(70), np.sin(70), np.cos(70), np.sin(70)]),
            "func": pr.quartic_1_func,
            "grad": pr.quartic_1_grad,
            "hess": pr.quartic_1_Hess,
        },
        {
            "name": "P6_quartic_2",
            "n": 4,
            "x0": np.array([np.cos(70), np.sin(70), np.cos(70), np.sin(70)]),
            "func": pr.quartic_2_func,
            "grad": pr.quartic_2_grad,
            "hess": pr.quartic_2_Hess,
        },
        {
            "name": "P7_Extended_Rosenbrock_n2",
            "n": 2,
            "x0": np.array([-1.2, 1.0]),
            "func": pr.ExtRF,
            "grad": pr.grad_ExtRF,
            "hess": pr.hess_ExtRF,
        },
        {
            "name": "P8_Extended_Rosenbrock_n100",
            "n": 100,
            "x0": np.concatenate(([ -1.2 ], np.ones(99))),
            "func": pr.ExtRF_100,
            "grad": pr.grad_ExtRF_100,
            "hess": pr.hess_ExtRF_100,
        },
        {
            "name": "P9_Beale",
            "n": 2,
            "x0": np.array([1.0, 1.0]),
            "func": pr.Beale_function,
            "grad": pr.Beale_gradient,
            "hess": pr.Beale_hessian,
        },
        {
            "name": "P10_exponential_10",
            "n": 10,
            "x0": np.concatenate(([1.0], np.zeros(9))),
            "func": pr.exponential_10_func,
            "grad": pr.exponential_10_grad,
            "hess": pr.exponential_10_Hess,
        },
        {
            "name": "P11_exponential_1000",
            "n": 1000,
            "x0": np.concatenate(([1.0], np.zeros(999))),
            "func": pr.exponential_1000_func,
            "grad": pr.exponential_1000_grad,
            "hess": pr.exponential_1000_Hess,
        },
        {
            "name": "P12_genhumps_5",
            "n": 5,
            "x0": np.full(5, -506.2),
            "func": pr.genhumps_5_func,
            "grad": pr.genhumps_5_grad,
            "hess": pr.genhumps_5_Hess,
        },
    ]

     # Define methods
    methods = [
        ("SD_armijo", lambda p, c1, c2, alpha: steepest_descent_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha)),
        ("SD_wolfe", lambda p, c1, c2, alpha: steepest_descent_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha)),
        ("Newton_armijo", lambda p, c1, c2, alpha: newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2, alpha)),
        ("Newton_wolfe", lambda p, c1, c2, alpha: newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], False, c1, c2, alpha)),
        ("Modified_Newton_armijo", lambda p, c1, c2, alpha: modified_newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2, alpha)),
        ("Modified_Newton_wolfe", lambda p, c1, c2, alpha: modified_newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], False, c1, c2, alpha)),
        ("BFGS_armijo", lambda p, c1, c2, alpha: BFGS_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha)),
        ("BFGS_wolfe", lambda p, c1, c2, alpha: BFGS_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha)),
        ("DFP_armijo", lambda p, c1, c2, alpha: DFP_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha)),
        ("DFP_wolfe", lambda p, c1, c2, alpha: DFP_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha)),
        ("L_BFGS_armijo", lambda p, c1, c2, alpha: L_BFGS_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha)),
        ("L_BFGS_wolfe", lambda p, c1, c2, alpha: L_BFGS_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha)),
        ("Newton_CG_armijo", lambda p, c1, c2, alpha: Newton_CG_param(p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2, alpha)),
        ("Newton_CG_wolfe", lambda p, c1, c2, alpha: Newton_CG_param(p["x0"], p["func"], p["grad"], p["hess"], False, c1, c2, alpha)),
    ]
    

    # for i, problem in enumerate(problems, 1):
    for method_name, method_func in methods:
        results = {}
        print(f"ALGO NAME: {method_name}")

        # for method_name, method_func in methods:
        for i, problem in enumerate(problems, 1):
            x0 = problem["x0"] # initial va; 
            n = len(x0)
            n = problem["n"]

            start_time = time.time()
            f_final, iters, gradient_norm, armijo, total_func_evals = method_func(problem)
            elapsed_time = time.time() - start_time

            results[problem['name']] = ({
                "Problem": f"P{i} ({problem['name']})",
                "Method": method_name,
                "f_final" : f_final,
                "Iterations": iters, 
                "Gradient_Norm": gradient_norm,
                "Time": elapsed_time,
                "Armijo": armijo, 
            })

       
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))

        all_problems = [problem['name'] for i, problem in enumerate(problems, 1)]
        iterations = [results[problem]['Iterations'] for problem in all_problems]
        axes[0].set_title(f"{method_name} Problem vs. Iterations")
        axes[0].set_xlabel("Problem")
        axes[0].set_xticks(np.arange(0, len(all_problems)))
        axes[0].set_xticklabels(all_problems, rotation=90)
        axes[0].set_ylabel("Iterations")
        axes[0].scatter(all_problems, iterations)

    
        gradient_norms = [results[problem]['Gradient_Norm'] for problem in all_problems]
        axes[1].set_title(f"{method_name} Norm of Gradient vs. Iterations")
        axes[1].set_xlabel("Norm of Gradient")
        axes[1].set_xticks(np.arange(0, len(all_problems)))
        # axes[1].set_xticklabels(gradient_norms, rotation=90)
        axes[1].set_ylabel("Iterations")
        axes[1].set_xticklabels(gradient_norms, rotation=90) 
        axes[1].scatter(np.arange(0, len(all_problems)), iterations) 
        plt.tight_layout() # to prevent plots from overlapping 
        plt.savefig(f"{method_name}.png")



TRIAL_TIMEOUT = 120 # 2 minutes 
import signal 
# Timeout handling for long-running trials
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Trial exceeded time limit")

def run_with_timeout(func, timeout_seconds=TRIAL_TIMEOUT):
    """
    Run a function with a timeout. If it takes longer than timeout_seconds,
    raise TimeoutException and allow the script to continue.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func()
        signal.alarm(0)  # Cancel alarm if completed
        return result, False  # result, timed_out
    except TimeoutException:
        signal.alarm(0)  # Cancel alarm
        return None, True  # None, timed_out

def compare_algorithms_csv(): 
    problems = [
        {
            "name": "P1_quad_10_10",
            "n": 10,
            "x0": 20 * np.random.rand(10) - 10,
            "func": pr.quad_10_10_func,
            "grad": pr.quad_10_10_grad,
            "hess": pr.quad_10_10_Hess,
        },
        {
            "name": "P2_quad_10_1000",
            "n": 10,
            "x0": 20 * np.random.rand(10) - 10,
            "func": pr.quad_10_1000_func,
            "grad": pr.quad_10_1000_grad,
            "hess": pr.quad_10_1000_Hess,
        },
        {
            "name": "P3_quad_1000_10", # I think this is causing problems 
            "n": 1000,
            "x0": 20 * np.random.rand(1000) - 10,
            "func": pr.quad_1000_10_func,
            "grad": pr.quad_1000_10_grad,
            "hess": pr.quad_1000_10_Hess,
        },
        {
            "name": "P4_quad_1000_1000",
            "n": 1000,
            "x0": 20 * np.random.rand(1000) - 10,
            "func": pr.quad_1000_1000_func,
            "grad": pr.quad_1000_1000_grad,
            "hess": pr.quad_1000_1000_Hess,
        },
        {
            "name": "P5_quartic_1",
            "n": 4,
            "x0": np.array([np.cos(70), np.sin(70), np.cos(70), np.sin(70)]),
            "func": pr.quartic_1_func,
            "grad": pr.quartic_1_grad,
            "hess": pr.quartic_1_Hess,
        },
        {
            "name": "P6_quartic_2",
            "n": 4,
            "x0": np.array([np.cos(70), np.sin(70), np.cos(70), np.sin(70)]),
            "func": pr.quartic_2_func,
            "grad": pr.quartic_2_grad,
            "hess": pr.quartic_2_Hess,
        },
        {
            "name": "P7_Extended_Rosenbrock_n2",
            "n": 2,
            "x0": np.array([-1.2, 1.0]),
            "func": pr.ExtRF,
            "grad": pr.grad_ExtRF,
            "hess": pr.hess_ExtRF,
        },
        {
            "name": "P8_Extended_Rosenbrock_n100",
            "n": 100,
            "x0": np.concatenate(([ -1.2 ], np.ones(99))),
            "func": pr.ExtRF_100,
            "grad": pr.grad_ExtRF_100,
            "hess": pr.hess_ExtRF_100,
        },
        {
            "name": "P9_Beale",
            "n": 2,
            "x0": np.array([1.0, 1.0]),
            "func": pr.Beale_function,
            "grad": pr.Beale_gradient,
            "hess": pr.Beale_hessian,
        },
        {
            "name": "P10_exponential_10",
            "n": 10,
            "x0": np.concatenate(([1.0], np.zeros(9))),
            "func": pr.exponential_10_func,
            "grad": pr.exponential_10_grad,
            "hess": pr.exponential_10_Hess,
        },
        {
            "name": "P11_exponential_1000",
            "n": 1000,
            "x0": np.concatenate(([1.0], np.zeros(999))),
            "func": pr.exponential_1000_func,
            "grad": pr.exponential_1000_grad,
            "hess": pr.exponential_1000_Hess,
        },
        {
            "name": "P12_genhumps_5",
            "n": 5,
            "x0": np.full(5, -506.2),
            "func": pr.genhumps_5_func,
            "grad": pr.genhumps_5_grad,
            "hess": pr.genhumps_5_Hess,
        },
    ]

     # Define methods
    methods = [
        ("SD_armijo", lambda p: steepest_descent_param(p["x0"], p["func"], p["grad"], True, C1, C2, START_ALPHA)),
        ("SD_wolfe", lambda p: steepest_descent_param(p["x0"], p["func"], p["grad"], False, C1, C2, START_ALPHA)),
        ("Newton_armijo", lambda p: newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], True, C1, C2, START_ALPHA)),
        ("Newton_wolfe", lambda p: newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], False, C1, C2, START_ALPHA)),
        ("Modified_Newton_armijo", lambda p: modified_newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], True, C1, C2, START_ALPHA)),
        ("Modified_Newton_wolfe", lambda p: modified_newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], False, C1, C2, START_ALPHA)),
        ("BFGS_armijo", lambda p: BFGS_param(p["x0"], p["func"], p["grad"], True, C1, C2, START_ALPHA)),
        ("BFGS_wolfe", lambda p: BFGS_param(p["x0"], p["func"], p["grad"], False, C1, C2, START_ALPHA)),
        ("DFP_armijo", lambda p: DFP_param(p["x0"], p["func"], p["grad"], True, C1, C2, START_ALPHA)),
        ("DFP_wolfe", lambda p: DFP_param(p["x0"], p["func"], p["grad"], False, C1, C2, START_ALPHA)),
        ("L_BFGS_armijo", lambda p: L_BFGS_param(p["x0"], p["func"], p["grad"], True, C1, C2, START_ALPHA)),
        ("L_BFGS_wolfe", lambda p: L_BFGS_param(p["x0"], p["func"], p["grad"], False, C1, C2, START_ALPHA)),
        ("Newton_CG_armijo", lambda p: Newton_CG_param(p["x0"], p["func"], p["grad"], p["hess"], True, C1, C2, START_ALPHA)),
        ("Newton_CG_wolfe", lambda p: Newton_CG_param(p["x0"], p["func"], p["grad"], p["hess"], False, C1, C2, START_ALPHA)),
    ]
    

    # for i, problem in enumerate(problems, 1):
    all_results = []
    for method_name, method_func in methods:
        
        print(f"ALGO NAME: {method_name}")
        is_armijo = "armijo" in method_name.lower()

        # for method_name, method_func in methods:
        for i, problem in enumerate(problems, 1):
            x0 = problem["x0"] # initial va; 
            n = len(x0)
            n = problem["n"]

            try:
                # Run the optimization with timeout
                start_time = time.time()
                
                # Wrap the method call in timeout
                result_data, timed_out = run_with_timeout(
                    lambda: method_func(problem),
                    timeout_seconds=TRIAL_TIMEOUT
                )

                elapsed_time = time.time() - start_time 

                if timed_out:
                    print(f"TIMEOUT ({TRIAL_TIMEOUT}s) - marked as failed")
                    result = {
                        'problem': problem['name'],
                        'method': method_name,
                        'iterations': K_MAX,
                        'converged': False,
                        'f_final': None,
                        'grad_norm': None,
                        'func_evals': None,
                        'time': elapsed_time,
                        'linesearch': 'Armijo' if is_armijo else 'Wolfe',
                        'timeout': True
                    }
                else:
                    f_final, iters, grad_norm, converged, func_evals = result_data
                    
                    # Store results
                    result = {
                        'problem': problem['name'],
                        'method': method_name,
                        'iterations': iters,
                        'converged': converged,
                        'f_final': f_final,
                        'grad_norm': grad_norm,
                        'func_evals': func_evals,
                        'time': elapsed_time,
                        'linesearch': 'Armijo' if is_armijo else 'Wolfe',
                        'timeout': False
                    }
                
                    status = "Converged" if converged else "Failed"
                    print(f"{status} in {iters} iters, {elapsed_time:.2f}s")
            
                    all_results.append(result)
            
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                result = {
                    'problem': problem['name'],
                    'method': method_name,
                    'iterations': K_MAX,
                    'converged': False,
                    'f_final': None,
                    'grad_norm': None,
                    'func_evals': None,
                    'time': None,
                    'linesearch': 'Armijo' if is_armijo else 'Wolfe',
                    'error': str(e)
                }
                all_results.append(result)
        
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save complete results to CSV
    csv_path = f"full_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\n{'='*80}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*80}")
    
    return df 
    



if __name__ == "__main__":
    compare_algorithms_csv()
    