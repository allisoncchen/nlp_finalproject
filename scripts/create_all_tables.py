import math 
import numpy as np 
import sys
import time 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scripts.problems as pr
import signal

K_MAX = 2000  # Maximum outer iterations
K_MAX_LINE_SEARCH = 100  # prevent excessive backtracking
TRIAL_TIMEOUT = 5 
TAU = 0.5  # Default tau value
BETA = 0.0004
EMIN = 10e-8

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

# Armijo version that takes c1 and tau
def Armijo_backtracking_param(x, func, gradient, p, alpha_init, c1, tau): 
    i = 0 
    a = alpha_init 
    
    while i < K_MAX_LINE_SEARCH:  # Changed from K_MAX to K_MAX_LINE_SEARCH 
        new_val = func(x + (a * p))
        modeled_val = func(x) + (c1 * a * float(np.dot(gradient(x).T, p)))

        # termination condition 
        if (new_val <= modeled_val): 
            return a, i + 1  # Return alpha and number of evaluations

        # Increment alpha 
        else: 
            a = tau * a # a is getting smaller 
        
        i += 1
    
    return -100, i


# wolfe now takes c1 and c2
def Wolfe_linesearch_param(x, p, func, gradient, alpha_init, c1, c2): 

    al = 0 
    au = float('inf') 
    a = alpha_init
    i = 0

    while i < K_MAX_LINE_SEARCH:  # Changed from K_MAX to K_MAX_LINE_SEARCH
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

# Runs the steepest descent algorithm with Armijo backtracking 
# Modified to accept c1, c2, start_alpha, tau as parameters
def steepest_descent_param(x, func, gradient, armijo, c1, c2, start_alpha, tau): 
    
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
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, alpha, c1, tau)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, alpha, c1, c2)

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
# Modified to accept c1, c2, start_alpha, tau as parameters
def newtons_method_param(x, func, gradient, hessian, armijo, c1, c2, start_alpha, tau):
    
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
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, start_alpha, c1, tau)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, start_alpha, c1, c2)
        
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
# Modified to accept c1, c2, start_alpha, tau as parameters
def modified_newtons_method_param(x, func, gradient, hessian, armijo, c1, c2, start_alpha, tau): 

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
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, start_alpha, c1, tau) # Choose using Armijo backtracking line search 
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, start_alpha, c1, c2) # Choose using Wolfe line search 
        
        total_func_evals += evals

        if alpha == -100: 
            return func(cur_vector), k, cur_gradient, False, total_func_evals
        
        # Update the iterate 
        cur_vector = cur_vector + (alpha * p)

        # Set k <- k + 1
        k += 1 
    
    return func(cur_vector), k, cur_gradient, False, total_func_evals


# Modified BFGS
def BFGS_param(x, func, gradient, armijo, c1, c2, start_alpha, tau): 
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
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, start_alpha, c1, tau)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, start_alpha, c1, c2)

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
# Modified to accept c1, c2, start_alpha, tau as parameters
def DFP_param(x, func, gradient, armijo, c1, c2, start_alpha, tau): 
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
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, start_alpha, c1, tau)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, start_alpha, c1, c2)

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
def L_BFGS_param(x, func, gradient, armijo, c1, c2, start_alpha, tau): 
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
            alpha, evals = Armijo_backtracking_param(cur_vector, func, gradient, p, start_alpha, c1, tau)
        else: 
            alpha, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient, start_alpha, c1, c2)

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
# Modified to accept c1, c2, start_alpha, tau as parameters
def Newton_CG_param(x, func, gradient_func, hessian, armijo, c1, c2, start_alpha, tau): 
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
            alpha_k, evals = Armijo_backtracking_param(cur_vector, func, gradient_func, p, start_alpha, c1, tau)
        else: 
            alpha_k, evals = Wolfe_linesearch_param(cur_vector, p, func, gradient_func, start_alpha, c1, c2)

        total_func_evals += evals

        if alpha_k == -100:
            return func(cur_vector), k, cur_gradient, False, total_func_evals

        cur_vector = cur_vector + (alpha_k * p)
        k += 1

    return func(cur_vector), k, cur_gradient, False, total_func_evals

def parameter_search():
    """
    Main function to run parameter search across all combinations
    """
    
    # Define parameter ranges to test
    c1_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.5]
    c2_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 1.9, 2.0]
    tau = TAU  # Fixed tau for main parameter search

    problems = [
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
        }
    ]
    
    methods = [
        ("SD_armijo", lambda p, c1, c2, alpha, tau: steepest_descent_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha, tau)),
        ("SD_wolfe", lambda p, c1, c2, alpha, tau: steepest_descent_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha, tau)),
        ("Newton_armijo", lambda p, c1, c2, alpha, tau: newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2, alpha, tau)),
        ("Newton_wolfe", lambda p, c1, c2, alpha, tau: newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], False, c1, c2, alpha, tau)),
        ("Modified_Newton_armijo", lambda p, c1, c2, alpha, tau: modified_newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2, alpha, tau)),
        ("Modified_Newton_wolfe", lambda p, c1, c2, alpha, tau: modified_newtons_method_param(p["x0"], p["func"], p["grad"], p["hess"], False, c1, c2, alpha, tau)),
        ("BFGS_armijo", lambda p, c1, c2, alpha, tau: BFGS_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha, tau)),
        ("BFGS_wolfe", lambda p, c1, c2, alpha, tau: BFGS_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha, tau)),
        ("DFP_armijo", lambda p, c1, c2, alpha, tau: DFP_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha, tau)),
        ("DFP_wolfe", lambda p, c1, c2, alpha, tau: DFP_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha, tau)),
        ("L_BFGS_armijo", lambda p, c1, c2, alpha, tau: L_BFGS_param(p["x0"], p["func"], p["grad"], True, c1, c2, alpha, tau)),
        ("L_BFGS_wolfe", lambda p, c1, c2, alpha, tau: L_BFGS_param(p["x0"], p["func"], p["grad"], False, c1, c2, alpha, tau)),
        ("Newton_CG_armijo", lambda p, c1, c2, alpha, tau: Newton_CG_param(p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2, alpha, tau)),
        ("Newton_CG_wolfe", lambda p, c1, c2, alpha, tau: Newton_CG_param(p["x0"], p["func"], p["grad"], p["hess"], False, c1, c2, alpha, tau)),
    ]
    
    # Create output directory
    output_dir = Path("parameter_search_results")
    output_dir.mkdir(exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Run parameter search
    total_runs = len(problems) * len(methods) * len(c1_values) * len(c2_values) * len(alpha_values)
    current_run = 0
    
    print(f"Starting parameter search: {total_runs} total runs")
    print(f"Problems: {len(problems)}, Methods: {len(methods)}")
    print(f"C1 values: {c1_values}")
    print(f"C2 values: {c2_values}")
    print(f"Alpha values: {alpha_values}")
    print(f"Tau (fixed): {tau}")
    print("-" * 80)
    
    for problem in problems:
        print(f"\n{'='*80}")
        print(f"Testing Problem: {problem['name']}")
        print(f"{'='*80}")
        
        for method_name, method_func in methods:
            print(f"\n  Method: {method_name}")
            
            # Determine if this is Armijo or Wolfe
            is_armijo = "armijo" in method_name.lower()
            
            # For Armijo methods, only vary c1 and alpha (c2 is not used)
            # For Wolfe methods, vary c1, c2, and alpha
            if is_armijo:
                param_combinations = list(itertools.product(c1_values, [0.9], alpha_values))
            else:
                param_combinations = list(itertools.product(c1_values, c2_values, alpha_values))
            
            for c1, c2, alpha in param_combinations:
                current_run += 1
                
                # Skip invalid combinations (c1 must be less than c2 for Wolfe)
                if not is_armijo and c1 >= c2:
                    print(f"    [{current_run}/{total_runs}] Skipping c1={c1:.4f}, c2={c2:.2f}, alpha={alpha:.1f} (c1 >= c2)")
                    continue
                
                print(f"    [{current_run}/{total_runs}] Testing c1={c1:.4f}, c2={c2:.2f}, alpha={alpha:.1f}...", end=" ")
                
                try:
                    # Run the optimization with timeout
                    start_time = time.time()
                    
                    # Wrap the method call in timeout
                    result_data, timed_out = run_with_timeout(
                        lambda: method_func(problem, c1, c2, alpha, tau),
                        timeout_seconds=TRIAL_TIMEOUT
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    if timed_out:
                        print(f"TIMEOUT ({TRIAL_TIMEOUT}s) - marked as failed")
                        result = {
                            'problem': problem['name'],
                            'method': method_name,
                            'c1': c1,
                            'c2': c2 if not is_armijo else None,
                            'alpha': alpha,
                            'tau': tau if is_armijo else None,
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
                            'c1': c1,
                            'c2': c2 if not is_armijo else None,
                            'alpha': alpha,
                            'tau': tau if is_armijo else None,
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
                        'c1': c1,
                        'c2': c2 if not is_armijo else None,
                        'alpha': alpha,
                        'tau': tau if is_armijo else None,
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
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\n{'='*80}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*80}")
    
    return df


def tau_search(main_results_df):
    """
    Separate focused search varying only tau
    Uses OPTIMAL c1 and alpha values from main parameter search for each method and problem
    """
    
    print("\n\n" + "="*80)
    print("STARTING TAU PARAMETER SEARCH (Armijo methods only)")
    print("="*80)
    print("Using optimal c1 and alpha from main parameter search for each method+problem")
    
    # Filter to converged Armijo runs from main search
    converged_armijo = main_results_df[
        (main_results_df['converged'] == True) & 
        (main_results_df['linesearch'] == 'Armijo')
    ].copy()
    
    # Tau values to test
    tau_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    problems = [
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
        }
    ]
    
    # Create output directory
    output_dir = Path("parameter_search_results")
    output_dir.mkdir(exist_ok=True)
    
    # Store all results
    tau_results = []
    
    print(f"Testing {len(tau_values)} tau values: {tau_values}")
    print("-" * 80)
    
    for problem in problems:
        print(f"\n{'='*80}")
        print(f"Testing Problem: {problem['name']}")
        print(f"{'='*80}")
        
        # Filter data for this problem
        problem_data = converged_armijo[converged_armijo['problem'] == problem['name']]
        
        # Define methods
        armijo_methods = [
            "SD_armijo", "Newton_armijo", "Modified_Newton_armijo", 
            "BFGS_armijo", "DFP_armijo", "L_BFGS_armijo", "Newton_CG_armijo"
        ]
        
        for method_name in armijo_methods:
            print(f"\n  Method: {method_name}")
            
            # Find optimal c1 and alpha for this method on this problem
            method_data = problem_data[problem_data['method'] == method_name]
            
            if len(method_data) == 0:
                print(f"    No converged data for {method_name} on {problem['name']}, skipping")
                continue
            
            # Find parameters that gave minimum iterations
            grouped = method_data.groupby(['c1', 'alpha'])['iterations'].mean()
            best_params = grouped.idxmin()
            c1_optimal = best_params[0]
            alpha_optimal = best_params[1]
            c2_default = 0.9  # Not used in Armijo but needed for function signature
            
            print(f"    Using optimal: c1={c1_optimal:.4f}, alpha={alpha_optimal:.2f}")
            
            # Create method lambda with optimal parameters
            if method_name == "SD_armijo":
                method_func = lambda p, tau, c1=c1_optimal, alpha=alpha_optimal: steepest_descent_param(
                    p["x0"], p["func"], p["grad"], True, c1, c2_default, alpha, tau)
            elif method_name == "Newton_armijo":
                method_func = lambda p, tau, c1=c1_optimal, alpha=alpha_optimal: newtons_method_param(
                    p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2_default, alpha, tau)
            elif method_name == "Modified_Newton_armijo":
                method_func = lambda p, tau, c1=c1_optimal, alpha=alpha_optimal: modified_newtons_method_param(
                    p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2_default, alpha, tau)
            elif method_name == "BFGS_armijo":
                method_func = lambda p, tau, c1=c1_optimal, alpha=alpha_optimal: BFGS_param(
                    p["x0"], p["func"], p["grad"], True, c1, c2_default, alpha, tau)
            elif method_name == "DFP_armijo":
                method_func = lambda p, tau, c1=c1_optimal, alpha=alpha_optimal: DFP_param(
                    p["x0"], p["func"], p["grad"], True, c1, c2_default, alpha, tau)
            elif method_name == "L_BFGS_armijo":
                method_func = lambda p, tau, c1=c1_optimal, alpha=alpha_optimal: L_BFGS_param(
                    p["x0"], p["func"], p["grad"], True, c1, c2_default, alpha, tau)
            elif method_name == "Newton_CG_armijo":
                method_func = lambda p, tau, c1=c1_optimal, alpha=alpha_optimal: Newton_CG_param(
                    p["x0"], p["func"], p["grad"], p["hess"], True, c1, c2_default, alpha, tau)
            else:
                continue
            
            for tau in tau_values:
                print(f"    Testing tau={tau:.1f}...", end=" ")
                
                try:
                    # Run the optimization with timeout
                    start_time = time.time()
                    
                    # Wrap the method call in timeout
                    result_data, timed_out = run_with_timeout(
                        lambda: method_func(problem, tau),
                        timeout_seconds=TRIAL_TIMEOUT
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    if timed_out:
                        print(f"⏱ TIMEOUT ({TRIAL_TIMEOUT}s) - marked as failed")
                        result = {
                            'problem': problem['name'],
                            'method': method_name,
                            'c1': c1_optimal,
                            'c2': None,
                            'alpha': alpha_optimal,
                            'tau': tau,
                            'iterations': K_MAX,
                            'converged': False,
                            'f_final': None,
                            'grad_norm': None,
                            'func_evals': None,
                            'time': elapsed_time,
                            'timeout': True
                        }
                    else:
                        f_final, iters, grad_norm, converged, func_evals = result_data
                        
                        # Store results
                        result = {
                            'problem': problem['name'],
                            'method': method_name,
                            'c1': c1_optimal,
                            'c2': None,
                            'alpha': alpha_optimal,
                            'tau': tau,
                            'iterations': iters,
                            'converged': converged,
                            'f_final': f_final,
                            'grad_norm': grad_norm,
                            'func_evals': func_evals,
                            'time': elapsed_time,
                            'timeout': False
                        }
                        
                        status = "Converged" if converged else "Failed"
                        print(f"{status} in {iters} iters, {elapsed_time:.2f}s")
                    
                    tau_results.append(result)
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    result = {
                        'problem': problem['name'],
                        'method': method_name,
                        'c1': c1_optimal,
                        'c2': None,
                        'alpha': alpha_optimal,
                        'tau': tau,
                        'iterations': K_MAX,
                        'converged': False,
                        'f_final': None,
                        'grad_norm': None,
                        'func_evals': None,
                        'time': None,
                        'error': str(e)
                    }
                    tau_results.append(result)
    
    # Convert to DataFrame
    tau_df = pd.DataFrame(tau_results)
    
    # Save tau results to CSV
    csv_path = output_dir / "tau_search_results.csv"
    tau_df.to_csv(csv_path, index=False)
    print(f"\n\n{'='*80}")
    print(f"Tau search results saved to: {csv_path}")
    print(f"{'='*80}")
    
    # Generate tau plots
    generate_tau_plots(tau_df, output_dir)
    
    return tau_df


def generate_tau_plots(tau_df, output_dir):
    """Generate plots showing effect of tau parameter"""
    
    print("\n" + "="*80)
    print("GENERATING TAU PLOTS")
    print("="*80)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Filter to converged runs
    converged_df = tau_df[tau_df['converged'] == True].copy()
    
    if len(converged_df) == 0:
        print("No converged runs for tau search!")
        return
    
    # Plot: Iterations vs Tau for each method
    print("Creating Tau effect plots for each method...")
    for method in converged_df['method'].unique():
        method_df = converged_df[converged_df['method'] == method]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by tau and problem
        for problem in method_df['problem'].unique():
            problem_df = method_df[method_df['problem'] == problem]
            grouped = problem_df.groupby('tau')['iterations'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=problem, linewidth=2, markersize=8)
        
        ax.set_xlabel('Tau Value', fontsize=12)
        ax.set_ylabel('Average Iterations', fontsize=12)
        ax.set_title(f'{method}: Effect of Tau on Iterations', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_path = plots_dir / f"{method}_tau_effect.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path.name}")
    
    # Combined plot: All methods on one plot
    print("\nCreating combined tau effect plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for method in converged_df['method'].unique():
        method_df = converged_df[converged_df['method'] == method]
        grouped = method_df.groupby('tau')['iterations'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Tau Value', fontsize=12)
    ax.set_ylabel('Average Iterations', fontsize=12)
    ax.set_title('Effect of Tau on Iterations (All Armijo Methods)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plot_path = plots_dir / "tau_effect_combined.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path.name}")
    
    # Find optimal tau for each method
    print("\n" + "="*80)
    print("OPTIMAL TAU VALUES BY METHOD")
    print("="*80)
    
    for method in converged_df['method'].unique():
        method_df = converged_df[converged_df['method'] == method]
        grouped = method_df.groupby('tau')['iterations'].mean()
        optimal_tau = grouped.idxmin()
        optimal_iters = grouped.min()
        print(f"{method}: tau = {optimal_tau:.1f} (avg {optimal_iters:.1f} iterations)")
    
    print(f"\nAll tau plots saved to: {plots_dir}")


def generate_summary_statistics(df, output_dir):
    """Generate summary statistics for the parameter search"""
    
    print("\n" + "="*80)
    print("GENERATING SUMMARY STATISTICS")
    print("="*80)
    
    # Overall success rate
    success_rate = df['converged'].mean() * 100
    print(f"\nOverall convergence rate: {success_rate:.1f}%")
    
    # Success rate by method
    print("\nConvergence rate by method:")
    method_success = df.groupby('method')['converged'].agg(['mean', 'count'])
    method_success['mean'] = method_success['mean'] * 100
    method_success.columns = ['Success Rate (%)', 'Total Runs']
    print(method_success.to_string())
    
    # Save method summary
    method_summary_path = output_dir / "method_summary.csv"
    method_success.to_csv(method_summary_path)
    print(f"\nMethod summary saved to: {method_summary_path}")
    
    # Best parameters for each method (based on average iterations for converged runs)
    print("\n" + "="*80)
    print("BEST PARAMETERS BY METHOD (minimum average iterations)")
    print("="*80)
    
    converged_df = df[df['converged'] == True]
    
    best_params = []
    for method in df['method'].unique():
        method_df = converged_df[converged_df['method'] == method]
        
        if len(method_df) > 0:
            if 'armijo' in method.lower():
                # For Armijo, group by c1 and alpha
                grouped = method_df.groupby(['c1', 'alpha'])['iterations'].agg(['mean', 'count'])
                if len(grouped) > 0:
                    best_idx = grouped['mean'].idxmin()
                    best_c1, best_alpha = best_idx
                    best_iters = grouped.loc[best_idx, 'mean']
                    best_count = grouped.loc[best_idx, 'count']
                    
                    best_params.append({
                        'method': method,
                        'c1': best_c1,
                        'c2': None,
                        'alpha': best_alpha,
                        'tau': TAU,
                        'avg_iterations': best_iters,
                        'num_converged': best_count
                    })
            else:
                # For Wolfe, group by c1, c2, and alpha
                grouped = method_df.groupby(['c1', 'c2', 'alpha'])['iterations'].agg(['mean', 'count'])
                if len(grouped) > 0:
                    best_idx = grouped['mean'].idxmin()
                    best_c1, best_c2, best_alpha = best_idx
                    best_iters = grouped.loc[best_idx, 'mean']
                    best_count = grouped.loc[best_idx, 'count']
                    
                    best_params.append({
                        'method': method,
                        'c1': best_c1,
                        'c2': best_c2,
                        'alpha': best_alpha,
                        'tau': None,
                        'avg_iterations': best_iters,
                        'num_converged': best_count
                    })
    
    best_params_df = pd.DataFrame(best_params)
    print("\n" + best_params_df.to_string(index=False))
    
    # Save best parameters
    best_params_path = output_dir / "best_parameters.csv"
    best_params_df.to_csv(best_params_path, index=False)
    print(f"\nBest parameters saved to: {best_params_path}")


def generate_plots(df, output_dir):
    """Generate visualization plots for parameter search results - MATPLOTLIB ONLY"""
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Filter to converged runs for most plots
    converged_df = df[df['converged'] == True].copy()
    
    # plot iterations vs C1 for each method (Armijo) - SEPARATE BY PROBLEM
    print("Creating C1 parameter plots for Armijo methods...")
    armijo_df = converged_df[converged_df['linesearch'] == 'Armijo']
    
    if len(armijo_df) > 0:
        for method in armijo_df['method'].unique():
            method_df = armijo_df[armijo_df['method'] == method]
            
            # Create separate plot for each problem
            for problem in sorted(method_df['problem'].unique()):
                problem_df = method_df[method_df['problem'] == problem]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by c1 and alpha, plot average iterations
                for alpha in sorted(problem_df['alpha'].unique()):
                    alpha_df = problem_df[problem_df['alpha'] == alpha]
                    grouped = alpha_df.groupby('c1')['iterations'].mean()
                    if len(grouped) > 0:
                        ax.plot(grouped.index, grouped.values, marker='o', label=f'alpha={alpha}', linewidth=2)
                
                ax.set_xlabel('C1 Value', fontsize=12)
                ax.set_ylabel('Average Iterations', fontsize=12)
                ax.set_title(f'{method}: Effect of C1 on Iterations ({problem})', fontsize=14, fontweight='bold')
                ax.set_xscale('log')
                
                # Only show legend for lines that actually have data
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=10)
                
                ax.grid(True, alpha=0.3)
                
                # Clean problem name for filename
                problem_clean = problem.replace('_', '').replace(' ', '')
                plot_path = plots_dir / f"{method}_c1_effect_{problem_clean}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {plot_path.name}")
    
    # plot iterations vs C2 for each method (Wolfe) - SEPARATE BY PROBLEM
    print("\n Creating C2 parameter plots for Wolfe methods...")
    wolfe_df = converged_df[converged_df['linesearch'] == 'Wolfe']
    
    if len(wolfe_df) > 0:
        for method in wolfe_df['method'].unique():
            method_df = wolfe_df[wolfe_df['method'] == method]
            
            # Create separate plot for each problem
            for problem in sorted(method_df['problem'].unique()):
                problem_df = method_df[method_df['problem'] == problem]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by c2 and alpha, plot average iterations
                for alpha in sorted(problem_df['alpha'].unique()):
                    alpha_df = problem_df[problem_df['alpha'] == alpha]
                    grouped = alpha_df.groupby('c2')['iterations'].mean()
                    if len(grouped) > 0:
                        ax.plot(grouped.index, grouped.values, marker='o', label=f'alpha={alpha}', linewidth=2)
                
                ax.set_xlabel('C2 Value', fontsize=12)
                ax.set_ylabel('Average Iterations', fontsize=12)
                ax.set_title(f'{method}: Effect of C2 on Iterations ({problem})', fontsize=14, fontweight='bold')
                
                # Only show legend for lines that actually have data
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=10)
                
                ax.grid(True, alpha=0.3)
                
                # Clean problem name for filename
                problem_clean = problem.replace('_', '').replace(' ', '')
                plot_path = plots_dir / f"{method}_c2_effect_{problem_clean}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {plot_path.name}")

    # plot iterations vs c1 for each method (Wolfe) - SEPARATE BY PROBLEM
    print("\nCreating c1 parameter plots for Wolfe methods...")
    
    if len(wolfe_df) > 0:
        for method in wolfe_df['method'].unique():
            method_df = wolfe_df[wolfe_df['method'] == method]
            
            # Create separate plot for each problem
            for problem in sorted(method_df['problem'].unique()):
                problem_df = method_df[method_df['problem'] == problem]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by c1 and alpha, plot average iterations
                for alpha in sorted(problem_df['alpha'].unique()):
                    alpha_df = problem_df[problem_df['alpha'] == alpha]
                    grouped = alpha_df.groupby('c1')['iterations'].mean()
                    if len(grouped) > 0:
                        ax.plot(grouped.index, grouped.values, marker='o', label=f'alpha={alpha}', linewidth=2)
                
                ax.set_xlabel('C1 Value', fontsize=12)
                ax.set_ylabel('Average Iterations', fontsize=12)
                ax.set_title(f'{method}: Effect of C1 on Iterations ({problem})', fontsize=14, fontweight='bold')
                ax.set_xscale('log')
                
                # Only show legend for lines that actually have data
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=10)
                
                ax.grid(True, alpha=0.3)
                
                # Clean problem name for filename
                problem_clean = problem.replace('_', '').replace(' ', '')
                plot_path = plots_dir / f"{method}_c1_effect_{problem_clean}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {plot_path.name}")        
    
    # plot iterations vs alpha for each method
    print("\nCreating alpha parameter plots for all methods...")
    for method in converged_df['method'].unique():
        method_df = converged_df[converged_df['method'] == method]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by alpha and problem, plot average iterations
        for problem in sorted(method_df['problem'].unique()):
            problem_df = method_df[method_df['problem'] == problem]
            grouped = problem_df.groupby('alpha')['iterations'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=problem, linewidth=2, markersize=8)
        
        ax.set_xlabel('Initial Alpha Value', fontsize=12)
        ax.set_ylabel('Average Iterations', fontsize=12)
        ax.set_title(f'{method}: Effect of Initial Alpha on Iterations', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_path = plots_dir / f"{method}_alpha_effect.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path.name}")
    
    # C1 vs C2 heatmap for Wolfe methods (average iterations) - SIDE BY SIDE BY PROBLEM
    print("\ncreating heatmaps for Wolfe methods...")
    for method in wolfe_df['method'].unique():
        method_df = wolfe_df[wolfe_df['method'] == method]
        
        # Create figure with 2 horizontal subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        problems = sorted(method_df['problem'].unique())
        
        for idx, problem in enumerate(problems):
            problem_df = method_df[method_df['problem'] == problem]
            
            # Average over all alpha values for this problem
            pivot_data = problem_df.pivot_table(
                values='iterations', 
                index='c2', 
                columns='c1', 
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                ax = axes[idx]
                
                # Create heatmap using imshow - REVERSED colormap (red = bad/high iterations)
                im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
                
                # Set ticks
                ax.set_xticks(np.arange(len(pivot_data.columns)))
                ax.set_yticks(np.arange(len(pivot_data.index)))
                ax.set_xticklabels(pivot_data.columns)
                ax.set_yticklabels(pivot_data.index)
                
                # Rotate the tick labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Average Iterations', rotation=270, labelpad=20)
                
                # Add text annotations
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        text = ax.text(j, i, f'{pivot_data.values[i, j]:.1f}',
                                     ha="center", va="center", color="black", fontsize=8)
                
                ax.set_title(f'{problem}', fontsize=12, fontweight='bold')
                ax.set_xlabel('C1 Value', fontsize=11)
                ax.set_ylabel('C2 Value', fontsize=11)
        
        # Overall title
        fig.suptitle(f'{method}: C1 vs C2 Heatmap (Average Iterations)', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        plot_path = plots_dir / f"{method}_c1_c2_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path.name}")
    
    # Success rate heatmap
    print("\nCreating success rate plots...")
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        if 'armijo' in method.lower():
            # For Armijo: c1 vs alpha
            pivot_data = method_df.pivot_table(
                values='converged',
                index='alpha',
                columns='c1',
                aggfunc='mean'
            ) * 100  # Convert to percentage
            
            if not pivot_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create heatmap
                im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
                
                # Set ticks
                ax.set_xticks(np.arange(len(pivot_data.columns)))
                ax.set_yticks(np.arange(len(pivot_data.index)))
                ax.set_xticklabels(pivot_data.columns)
                ax.set_yticklabels(pivot_data.index)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Success Rate (%)', rotation=270, labelpad=20)
                
                # Add text annotations
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        text = ax.text(j, i, f'{pivot_data.values[i, j]:.0f}',
                                     ha="center", va="center", color="black", fontsize=10)
                
                ax.set_title(f'{method}: Success Rate (%) - C1 vs Alpha', fontsize=14, fontweight='bold')
                ax.set_xlabel('C1 Value', fontsize=12)
                ax.set_ylabel('Initial Alpha', fontsize=12)
                
                plot_path = plots_dir / f"{method}_success_rate.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {plot_path.name}")
        else:
            # For Wolfe: c1 vs c2 (averaged over alpha)
            pivot_data = method_df.pivot_table(
                values='converged',
                index='c2',
                columns='c1',
                aggfunc='mean'
            ) * 100
            
            if not pivot_data.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create heatmap
                im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
                
                # Set ticks
                ax.set_xticks(np.arange(len(pivot_data.columns)))
                ax.set_yticks(np.arange(len(pivot_data.index)))
                ax.set_xticklabels(pivot_data.columns)
                ax.set_yticklabels(pivot_data.index)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Success Rate (%)', rotation=270, labelpad=20)
                
                # Add text annotations
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        val = pivot_data.values[i, j]
                        if not np.isnan(val):
                            text = ax.text(j, i, f'{val:.0f}',
                                         ha="center", va="center", color="black", fontsize=10)
                
                ax.set_title(f'{method}: Success Rate (%) - C1 vs C2', fontsize=14, fontweight='bold')
                ax.set_xlabel('C1 Value', fontsize=12)
                ax.set_ylabel('C2 Value', fontsize=12)
                
                plot_path = plots_dir / f"{method}_success_rate.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {plot_path.name}")
    
    # Optimal parameters plot for Wolfe methods - SEPARATE BY PROBLEM
    print("\nCreating optimal parameter plots for Wolfe methods...")
    wolfe_methods = [m for m in converged_df['method'].unique() if 'wolfe' in m.lower()]
    
    if len(wolfe_methods) > 0:
        # Create separate plot for each problem
        for problem in sorted(converged_df['problem'].unique()):
            problem_df = converged_df[converged_df['problem'] == problem]
            
            # Get best parameters for each Wolfe method for this problem
            wolfe_best_params = []
            for method in wolfe_methods:
                method_df = problem_df[problem_df['method'] == method]
                if len(method_df) > 0:
                    grouped = method_df.groupby(['c1', 'c2', 'alpha'])['iterations'].mean()
                    best_idx = grouped.idxmin()
                    wolfe_best_params.append({
                        'method': method,
                        'c1': best_idx[0],
                        'c2': best_idx[1],
                        'alpha': best_idx[2]
                    })
            
            if len(wolfe_best_params) > 0:
                # Create figure with 3 subplots
                fig, axes = plt.subplots(3, 1, figsize=(12, 12))
                
                methods = [p['method'] for p in wolfe_best_params]
                c1_values = [p['c1'] for p in wolfe_best_params]
                c2_values = [p['c2'] for p in wolfe_best_params]
                alpha_values = [p['alpha'] for p in wolfe_best_params]
                
                x_pos = np.arange(len(methods))
                
                # optimal C1
                axes[0].bar(x_pos, c1_values, color='skyblue', edgecolor='navy', linewidth=1.5)
                axes[0].set_xticks(x_pos)
                axes[0].set_xticklabels(methods, rotation=45, ha='right')
                axes[0].set_ylabel('Optimal C1', fontsize=12)
                axes[0].set_yscale('log')
                axes[0].set_title(f'Optimal C1 Parameter by Method - Wolfe ({problem})', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')
                
                # optimal C2
                axes[1].bar(x_pos, c2_values, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
                axes[1].set_xticks(x_pos)
                axes[1].set_xticklabels(methods, rotation=45, ha='right')
                axes[1].set_ylabel('Optimal C2', fontsize=12)
                axes[1].set_title(f'Optimal C2 Parameter by Method - Wolfe ({problem})', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')
                
                # optimal alpha
                axes[2].bar(x_pos, alpha_values, color='lightcoral', edgecolor='darkred', linewidth=1.5)
                axes[2].set_xticks(x_pos)
                axes[2].set_xticklabels(methods, rotation=45, ha='right')
                axes[2].set_ylabel('Optimal Alpha', fontsize=12)
                axes[2].set_title(f'Optimal Initial Alpha by Method - Wolfe ({problem})', fontsize=14, fontweight='bold')
                axes[2].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                # Clean problem name for filename
                problem_clean = problem.replace('_', '').replace(' ', '')
                plot_path = plots_dir / f"optimal_params_wolfe_{problem_clean}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {plot_path.name}")
    
    # Optimal parameters plot for Armijo methods - SEPARATE BY PROBLEM
    armijo_methods = [m for m in converged_df['method'].unique() if 'armijo' in m.lower()]
    
    if len(armijo_methods) > 0:
        # Create separate plot for each problem
        for problem in sorted(converged_df['problem'].unique()):
            problem_df = converged_df[converged_df['problem'] == problem]
            
            # Get best parameters for each Armijo method for this problem
            armijo_best_params = []
            for method in armijo_methods:
                method_df = problem_df[problem_df['method'] == method]
                if len(method_df) > 0:
                    grouped = method_df.groupby(['c1', 'alpha'])['iterations'].mean()
                    best_idx = grouped.idxmin()
                    armijo_best_params.append({
                        'method': method,
                        'c1': best_idx[0],
                        'alpha': best_idx[1]
                    })
            
            if len(armijo_best_params) > 0:
                # Create figure with 2 subplots (vertically stacked)
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                methods = [p['method'] for p in armijo_best_params]
                c1_values = [p['c1'] for p in armijo_best_params]
                alpha_values = [p['alpha'] for p in armijo_best_params]
                
                x_pos = np.arange(len(methods))
                
                # optimal C1
                axes[0].bar(x_pos, c1_values, color='skyblue', edgecolor='navy', linewidth=1.5)
                axes[0].set_xticks(x_pos)
                axes[0].set_xticklabels(methods, rotation=45, ha='right')
                axes[0].set_ylabel('Optimal C1', fontsize=12)
                axes[0].set_yscale('log')
                axes[0].set_title(f'Optimal C1 Parameter by Method - Armijo ({problem})', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')
                
                # alpha
                axes[1].bar(x_pos, alpha_values, color='lightcoral', edgecolor='darkred', linewidth=1.5)
                axes[1].set_xticks(x_pos)
                axes[1].set_xticklabels(methods, rotation=45, ha='right')
                axes[1].set_ylabel('Optimal Alpha', fontsize=12)
                axes[1].set_title(f'Optimal Initial Alpha by Method - Armijo ({problem})', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                # Clean problem name for filename
                problem_clean = problem.replace('_', '').replace(' ', '')
                plot_path = plots_dir / f"optimal_params_armijo_{problem_clean}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {plot_path.name}")
    
    print(f"\nAll plots saved to: {plots_dir}")


def generate_final_method_comparison(main_results_df, tau_results_df, output_dir):
    """
    Generate final method comparison plot using OPTIMAL parameters:
    - Wolfe methods: Best (c1, c2, alpha) from main search
    - Armijo methods: Best (c1, alpha) from main search + Best tau from tau search
    """
    
    print("\n" + "="*80)
    print("GENERATING FINAL METHOD COMPARISON PLOT")
    print("="*80)
    print("Using optimal parameters from all searches")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Filter to converged runs
    converged_main = main_results_df[main_results_df['converged'] == True].copy()
    converged_tau = tau_results_df[tau_results_df['converged'] == True].copy()
    
    # Create vertically stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    problems = sorted(converged_main['problem'].unique())
    
    for idx, problem in enumerate(problems):
        ax = axes[idx]
        
        # Dictionary to store optimal iterations for each method
        method_optimal_iters = {}
        
        # Get all methods
        all_methods = converged_main['method'].unique()
        
        for method in all_methods:
            if 'armijo' in method.lower():
                # For Armijo: Use best tau from tau search (which already uses optimal c1, alpha)
                tau_method_data = converged_tau[
                    (converged_tau['problem'] == problem) &
                    (converged_tau['method'] == method)
                ]
                
                if len(tau_method_data) > 0:
                    # Find minimum iterations across all tau values
                    optimal_iters = tau_method_data['iterations'].min()
                    method_optimal_iters[method] = optimal_iters
            else:
                # For Wolfe: Use best (c1, c2, alpha) from main search
                main_method_data = converged_main[
                    (converged_main['problem'] == problem) &
                    (converged_main['method'] == method)
                ]
                
                if len(main_method_data) > 0:
                    # Group by c1, c2, alpha and find minimum
                    grouped = main_method_data.groupby(['c1', 'c2', 'alpha'])['iterations'].mean()
                    optimal_iters = grouped.min()
                    method_optimal_iters[method] = optimal_iters
        
        # Sort and plot
        if len(method_optimal_iters) > 0:
            method_comparison = pd.Series(method_optimal_iters).sort_values()
            
            y_pos = np.arange(len(method_comparison))
            ax.barh(y_pos, method_comparison.values, 
                   color='skyblue', edgecolor='navy', linewidth=1.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(method_comparison.index, fontsize=10)
            ax.set_xlabel('Iterations (Optimal Parameters)', fontsize=12)
            ax.set_title(f'Method Performance Comparison - {problem}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_path = plots_dir / "method_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path.name}")
    print(f"   Using Wolfe optimal: (c1, c2, alpha) from main search")
    print(f"   Using Armijo optimal: (c1, alpha) from main search + tau from tau search")


if __name__ == "__main__":

    # Run main parameter search (c1, c2, alpha with fixed tau)
    results_df = parameter_search()
    
    # Generate summary statistics and plots
    generate_summary_statistics(results_df, Path("parameter_search_results"))
    generate_plots(results_df, Path("parameter_search_results"))
    
    print("\n" + "="*80)
    print("MAIN PARAMETER SEARCH COMPLETE!")
    print("="*80)
    print(f"\nTotal runs: {len(results_df)}")
    print(f"Converged: {results_df['converged'].sum()}")
    print(f"Failed: {(~results_df['converged']).sum()}")
    
    # Run separate tau search using optimal parameters from main search
    tau_results_df = tau_search(results_df)
    
    # Generate final method comparison using optimal parameters from both searches
    generate_final_method_comparison(results_df, tau_results_df, Path("parameter_search_results"))
    
    print("\n" + "="*80)
    print("ALL SEARCHES COMPLETE!")
    print("="*80)
    print("\nCheck the 'parameter_search_results' directory for:")
    print("all_results.csv: Complete results table")
    print("tau_search_results.csv: Tau-specific search results")
    print("best_parameters.csv: Best parameters for each method")
    print("method_summary.csv: Success rates by method")
    print("plots/: All visualization plots")
    print("="*80)