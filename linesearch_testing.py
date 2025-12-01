import math 
import numpy as np 
import sys
import time 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import your existing code
import problems as pr

K_MAX = 2000
TAO = 0.5
BETA = 0.0004
EMIN = 10e-8


# ============================================================================
# PARAMETERIZED LINE SEARCH FUNCTIONS
# ============================================================================

# Armijo version that takes c1 
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


# wolfe now takes c1 and c2
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


# ============================================================================
# PARAMETER SEARCH FRAMEWORK
# ============================================================================

def parameter_search():
    """
    Main function to run parameter search across all combinations
    """
    
    # Define parameter ranges to test
    c1_values = [1e-4, 1e-3, 1e-2, 0.1]
    c2_values = [0.1, 0.5, 0.9, 0.99]
    alpha_values = [0.1, 0.5, 1.0, 2.0]
    
    # Define problems (start with a smaller subset, comment/uncomment as needed)
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
        # {
        #     "name": "P3_quad_1000_10",
        #     "n": 1000,
        #     "x0": 20 * np.random.rand(1000) - 10,
        #     "func": pr.quad_1000_10_func,
        #     "grad": pr.quad_1000_10_grad,
        #     "hess": pr.quad_1000_10_Hess,
        # },
        # {
        #     "name": "P4_quad_1000_1000",
        #     "n": 1000,
        #     "x0": 20 * np.random.rand(1000) - 10,
        #     "func": pr.quad_1000_1000_func,
        #     "grad": pr.quad_1000_1000_grad,
        #     "hess": pr.quad_1000_1000_Hess,
        # },
        {
            "name": "P5_quartic_1",
            "n": 4,
            "x0": np.array([np.cos(70), np.sin(70), np.cos(70), np.sin(70)]),
            "func": pr.quartic_1_func,
            "grad": pr.quartic_1_grad,
            "hess": pr.quartic_1_Hess,
        },
        # Add more problems as needed
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
                    # Run the optimization
                    start_time = time.time()
                    f_final, iters, grad_norm, converged, func_evals = method_func(problem, c1, c2, alpha)
                    elapsed_time = time.time() - start_time
                    
                    # Store results
                    result = {
                        'problem': problem['name'],
                        'method': method_name,
                        'c1': c1,
                        'c2': c2 if not is_armijo else None,
                        'alpha': alpha,
                        'iterations': iters,
                        'converged': converged,
                        'f_final': f_final,
                        'grad_norm': grad_norm,
                        'func_evals': func_evals,
                        'time': elapsed_time,
                        'linesearch': 'Armijo' if is_armijo else 'Wolfe'
                    }
                    
                    all_results.append(result)
                    
                    status = "Converged" if converged else "Failed"
                    print(f"{status} in {iters} iters, {elapsed_time:.2f}s")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    result = {
                        'problem': problem['name'],
                        'method': method_name,
                        'c1': c1,
                        'c2': c2 if not is_armijo else None,
                        'alpha': alpha,
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
    
    # Generate summary statistics
    generate_summary_statistics(df, output_dir)
    
    # Generate plots
    generate_plots(df, output_dir)
    
    return df


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
    
    # plot iterations vs C1 for each method (Armijo)
    print("Creating C1 parameter plots for Armijo methods...")
    armijo_df = converged_df[converged_df['linesearch'] == 'Armijo']
    
    if len(armijo_df) > 0:
        for method in armijo_df['method'].unique():
            method_df = armijo_df[armijo_df['method'] == method]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by c1 and alpha, plot average iterations
            for alpha in sorted(method_df['alpha'].unique()):
                alpha_df = method_df[method_df['alpha'] == alpha]
                grouped = alpha_df.groupby('c1')['iterations'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=f'alpha={alpha}', linewidth=2)
            
            ax.set_xlabel('C1 Value', fontsize=12)
            ax.set_ylabel('Average Iterations', fontsize=12)
            ax.set_title(f'{method}: Effect of C1 on Iterations', fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plot_path = plots_dir / f"{method}_c1_effect.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved: {plot_path.name}")
    
    # plot iterations vs C2 for each method (Wolfe)
    print("\n Creating C2 parameter plots for Wolfe methods...")
    wolfe_df = converged_df[converged_df['linesearch'] == 'Wolfe']
    
    if len(wolfe_df) > 0:
        for method in wolfe_df['method'].unique():
            method_df = wolfe_df[wolfe_df['method'] == method]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by c2 and alpha, plot average iterations
            for alpha in sorted(method_df['alpha'].unique()):
                alpha_df = method_df[method_df['alpha'] == alpha]
                grouped = alpha_df.groupby('c2')['iterations'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=f'alpha={alpha}', linewidth=2)
            
            ax.set_xlabel('C2 Value', fontsize=12)
            ax.set_ylabel('Average Iterations', fontsize=12)
            ax.set_title(f'{method}: Effect of C2 on Iterations', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plot_path = plots_dir / f"{method}_c2_effect.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved: {plot_path.name}")
    
    # plot iterations vs alpha for each method
    print("\nCreating alpha parameter plots for all methods...")
    for method in converged_df['method'].unique():
        method_df = converged_df[converged_df['method'] == method]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by alpha, plot average iterations
        grouped = method_df.groupby('alpha')['iterations'].agg(['mean', 'std', 'count'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                   marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xlabel('Initial Alpha Value', fontsize=12)
        ax.set_ylabel('Average Iterations', fontsize=12)
        ax.set_title(f'{method}: Effect of Initial Alpha on Iterations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plot_path = plots_dir / f"{method}_alpha_effect.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {plot_path.name}")
    
    # C1 vs C2 heatmap for Wolfe methods (average iterations)
    print("\ncreating heatmaps for Wolfe methods...")
    for method in wolfe_df['method'].unique():
        method_df = wolfe_df[wolfe_df['method'] == method]
        
        # Average over all alpha values
        pivot_data = method_df.pivot_table(
            values='iterations', 
            index='c2', 
            columns='c1', 
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap using imshow
            im = ax.imshow(pivot_data.values, cmap='YlOrRd_r', aspect='auto')
            
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
                                 ha="center", va="center", color="black", fontsize=10)
            
            ax.set_title(f'{method}: C1 vs C2 Heatmap (Average Iterations)', fontsize=14, fontweight='bold')
            ax.set_xlabel('C1 Value', fontsize=12)
            ax.set_ylabel('C2 Value', fontsize=12)
            
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
    
    # Comparison plot: Best parameters across methods
    print("\nCreating method comparison plots...")
    
    # Average iterations by method (for converged runs)
    fig, ax = plt.subplots(figsize=(12, 6))
    method_stats = converged_df.groupby('method')['iterations'].agg(['mean', 'std'])
    method_stats = method_stats.sort_values('mean')
    
    y_pos = np.arange(len(method_stats))
    ax.barh(y_pos, method_stats['mean'], xerr=method_stats['std'], 
           capsize=5, color='skyblue', edgecolor='navy', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_stats.index, fontsize=10)
    ax.set_xlabel('Average Iterations (Converged Runs)', fontsize=12)
    ax.set_title('Method Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plot_path = plots_dir / "method_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {plot_path.name}")
    
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":

    results_df = parameter_search()
    
    print("\n" + "="*80)
    print("PARAMETER SEARCH COMPLETE!")
    print("="*80)
    print(f"\nTotal runs: {len(results_df)}")
    print(f"Converged: {results_df['converged'].sum()}")
    print(f"Failed: {(~results_df['converged']).sum()}")
    print("\nCheck the 'parameter_search_results' directory for:")
    print("all_results.csv: Complete results table")
    print("best_parameters.csv: Best parameters for each method")
    print("method_summary.csv: Success rates by method")
    print("plots/: All visualization plots")
    print("="*80)