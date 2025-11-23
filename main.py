import math 
import numpy as np 
import sys
import time 
from fpdf import FPDF
import functions 


START_ALPHA = 1
C1 = 0.0004
C2 = 0.9
TAO = 0.5
K_MAX = 6000
BETA = 0.0004
EMIN = 10e-8

    
# Armijo condition – find optimal steplength (alpha) selection
def Armijo_backtracking(x, func, gradient, p, alpha_init): 
    i = 0 
    a = alpha_init 
    
    while i < K_MAX: 
        new_val = func(x + (a * p))
        modeled_val = func(x) + (C1 * a * float(np.dot(gradient(x).T, p)))

        # termination condition 
        if (new_val <= modeled_val): 
            return a


        # Increment alpha 
        else: 
            a = TAO * a # a is getting smaller 
        
        i += 1
    
    print("Couldn't find alpha value")
    return -100


# Runs Wolfe linesearch to identify the best alpha using Armijo and the curvature condition 
def Wolfe_linesearch(x, p, func, gradient): 

    al = 0 
    au = float('inf') 
    a = 1
    i = 0


    while i < K_MAX:
        new_val = func(x + (a * p))
        modeled_val = func(x) + (C1 * a * float(np.dot(gradient(x).T, p)))
        
        if (new_val > modeled_val): # armijo's
            au = a
        else: 
            if (np.dot((gradient(x + (a * p)).T), p) < C2 * np.dot(gradient(x).T, p)): # curvature condition 
                al = a
            else: 
                return a # stopping and returning alpha as output of linesearch 
        
        if au < float('inf'): 
            a = (al + au) / 2
        else: 
            a = 2 * a
        
        i += 1
        # print(i)
    
    print("made it out of the wolfe line search loop – probably shouldn't happen.")
    return a

# Runs the steepest descent algorithm with Armijo backtracking 
# Output: x
def steepest_descent(x, func, gradient, armijo): 
    
    alpha = START_ALPHA
    
    prev_vector = x
    cur_vector = x
    k = 0 # outer loop 
    stopping_point = (10**-8) * max(1, np.linalg.norm(gradient(x)))

    # Stopping Condition: hit max iterations without convergence
    while (k < K_MAX): 

        norm_cur_gradient_vector = np.linalg.norm(gradient(cur_vector))
        recent_val = func(cur_vector)
        print(f"Iteration: {k}, f(x): {recent_val}, ||gradf||: {norm_cur_gradient_vector}, alpha: {alpha}")

        # Stopping Condition: satisfiable convergence
        if norm_cur_gradient_vector <= stopping_point: 
            print("Final value of objective function: ", recent_val)
            print("CONVERGED\n")
            return func(cur_vector)
        
        # for every iteration after the initial, recalculate alpha 
        if k > 0: 
            theta = np.dot(gradient(prev_vector), (-1 * gradient(prev_vector).T))
            alpha = 2 * (func(cur_vector) - func(prev_vector)) / theta

        
        # choose a descent direction p  
        p = -gradient(cur_vector)

        # find alpha
        if armijo == True: 
            alpha = Armijo_backtracking(cur_vector, func, gradient, p, alpha)
        else: 
            alpha = Wolfe_linesearch(cur_vector, func, gradient, p, alpha)


        if alpha == -100: 
            print("Couldn't find alpha – likely an error")
            return -1
        
        # Update the the iterate 
        prev_vector = cur_vector 
        cur_vector = cur_vector + (alpha * p)


        # Set k <- k + 1
        k += 1 
    
    print("Final value of objective function: ", recent_val)
    print("HIT MAX ITERATIONS IN STEEPEST DESCENT")



# Runs Newton's method with Armijo backtracking to find alpha
# Output: x
def newtons_method(x, func, gradient, hessian, armijo):
    
    cur_vector = x
    k = 0 # for the outer loop 
    

    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector

    # Stopping Condition – hit max iterations without convergence
    while (k < K_MAX): 

        alpha = START_ALPHA
        cur_gradient = np.linalg.norm(gradient(cur_vector))
        recent_val = func(cur_vector)
        print(f"Iteration: {k}, f(x): {recent_val}, ||gradf||: {cur_gradient}, alpha: {alpha}")


        # Stopping Condition – satisfiable convergence
        if cur_gradient <= stopping_point: 
            print("Final value of objective function: ", recent_val)
            print("CONVERGED\n")
            return func(cur_vector)
    

        # Choose a descent direction p  
        gradient_vector = gradient(cur_vector)
        hessian_matrix = hessian(cur_vector)
        try: 
            p = np.linalg.solve(hessian_matrix, -gradient_vector)
        except np.linalg.LinAlgError:
            p = -gradient(cur_vector) # default to steepest descent 
            
        # Find alpha 
        if armijo == True: 
            alpha = Armijo_backtracking(cur_vector, func, gradient, p, 1)
        else: 
            alpha = Wolfe_linesearch(cur_vector, p, func, gradient)
        
        if alpha == -100: 
            print("Couldn't find alpha – likely an error")
            return -1
        
        
        # Update the the iterate 
        cur_vector = cur_vector + (alpha * p)


        # Set k <- k + 1
        k += 1 
    
    print("Final value of objective function: ", recent_val)
    print("HIT MAX ITERATIONS IN NEWTONS")


# Runs the cholesky algorithm to find the optimal addition to the hessian matrix
# Output: delta
def cholesky_with_added_multiple_of_identity(A):
    print("in here\n")
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
# Output: x
def modified_newtons_method(x, func, gradient, hessian, armijo): 

    cur_vector = x
    k = 0 # outer loop 
    delta = 0
    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    
    while (k < K_MAX): 
        alpha = START_ALPHA
        cur_gradient = np.linalg.norm(gradient(cur_vector))
        recent_val = func(cur_vector)
        print(f"Iteration: {k}, f(x): {recent_val}, ||gradf||: {cur_gradient}, alpha: {alpha}, delta {delta} ")
        
        if cur_gradient <= stopping_point: 
            print("Final value of objective function: ", recent_val)
            print("CONVERGED\n")
            return func(cur_vector)
    
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
            alpha = Armijo_backtracking(cur_vector, func, gradient, p, 1) # Choose using Armijo backtracking line search 
        else: 
            alpha = Wolfe_linesearch(cur_vector, p, func, gradient) # Choose using Armijo backtracking line search 
        

        if alpha == -100: 
            print("Couldn't find alpha – likely an error")
            return -1
        
        # Update the iterate 
        cur_vector = cur_vector + (alpha * p)

        # Set k <- k + 1
        k += 1 
    
    print("Final value of objective function: ", recent_val)
    print("HIT MAX ITERATIONS IN MODIFIED NEWTONS")



def BFGS(x, func, gradient, armijo): 
    length = len(x)
    H = np.identity(length)
    prev_vector = x
    cur_vector = x
    k = 0 
    I = np.identity(length)
    alpha = START_ALPHA

    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    
    
    if (len(x) == 10000): 
        recent_val, k = L_BFGS_large_memory(x, func, gradient)
        return recent_val, k
    else:  
        while (k < K_MAX): 
            gradient_vector = gradient(cur_vector)
            cur_gradient = np.linalg.norm(gradient(cur_vector))
            recent_val = func(cur_vector)
            print(f"Iteration: {k}, f(x): {recent_val}, ||gradf||: {cur_gradient}, alpha: {alpha}")

            # stopping Condition, satisfiable convergence
            if cur_gradient <= stopping_point: 
                print("Final value of objective function: ", recent_val)
                print("CONVERGED\n")
                return func(cur_vector), k
            
        
            p = -H @ gradient_vector
            
            # choose a steplength with wolfe's
            if armijo == True: 
                alpha = Armijo_backtracking(cur_vector, func, gradient, p, 1)
            else: 
                alpha = Wolfe_linesearch(cur_vector, p, func, gradient)

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
        
        
    print("Final value of objective function: ", recent_val)
    print("HIT MAX ITERATIONS IN BFGS")
    return recent_val, k


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


# Runs L_BFGS for question 7 based off of instructions from lecture 
def L_BFGS_large_memory(x, func, gradient): 
    gamma = 1 # starts off as one from the specs 
    n = len(x)
    m = max(n, 10) 
    I = np.identity(n)
    alpha = START_ALPHA
    k = 0 
    cur_vector = x.copy()
    prev_vector = x.copy()
    

    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    y_s_pairs = [] 

    while (k < K_MAX): 
        cur_gradient = gradient(cur_vector)
        cur_gradient_norm = np.linalg.norm(cur_gradient)
        recent_val = func(cur_vector)
        print(f"Iteration: {k}, f(x): {recent_val}, ||gradf||: {cur_gradient_norm}, alpha: {alpha}")


        # stopping Condition, satisfiable convergence
        if cur_gradient_norm <= stopping_point: 
            print("Final value of objective function: ", recent_val)
            print("CONVERGED\n")
            return func(cur_vector), k

        
        p = -(two_loop_recursion(cur_gradient, y_s_pairs, gamma, I)) # y_s_pairs is going to be empty on first iteration
        # print(p)

        # choose a steplength with wolfe's
        alpha = Wolfe_linesearch(cur_vector, p, func, gradient)

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
    
    print("Final value of objective function: ", recent_val)
    print("HIT MAX ITERATIONS IN L_BFGS")

    return recent_val, k


def L_BFGS(x, func, gradient, armijo): 
    gamma = 1 # starts off as one from the specs 
    n = len(x)
    m = min(n, 10) 
    I = np.identity(n)
    alpha = START_ALPHA
    k = 0 
    cur_vector = x.copy()
    prev_vector = x.copy()
    

    stopping_point = 1e-8 * max(1, np.linalg.norm(gradient(x))) # gradient of OG vector 
    y_s_pairs = [] 

    while (k < K_MAX): 
        cur_gradient = gradient(cur_vector)
        cur_gradient_norm = np.linalg.norm(cur_gradient)
        recent_val = func(cur_vector)
        print(f"Iteration: {k}, f(x): {recent_val}, ||gradf||: {cur_gradient_norm}, alpha: {alpha}")


        # stopping Condition, satisfiable convergence
        if cur_gradient_norm <= stopping_point: 
            print("Final value of objective function: ", recent_val)
            print("CONVERGED\n")
            return func(cur_vector), k

        
        p = -(two_loop_recursion(cur_gradient, y_s_pairs, gamma, I)) # y_s_pairs is going to be empty on first iteration
        

        # choose a steplength with wolfe's
        if armijo == True: 
            alpha = Armijo_backtracking(cur_vector, func, gradient, p, 1)
        else: 
            alpha = Wolfe_linesearch(cur_vector, p, func, gradient)

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
    
    print("Final value of objective function: ", recent_val)
    print("HIT MAX ITERATIONS IN L_BFGS")

    return recent_val, k


# Runs Newton CG method with Wolfe line search 
# Output: x
n = 0.01
def Newton_CG(x, func, gradient_func, hessian, armijo): 
    k = 0 
    alpha_k = START_ALPHA
    cur_vector = x
    
    while k < K_MAX: 
        z = 0 
        r = gradient_func(cur_vector)
        d = -r

        gradient = np.linalg.norm(gradient_func(cur_vector))
        func_val = func(cur_vector)
        p = 0.0
        stopping_point = 1e-8 * max(1, np.linalg.norm(gradient))
    
        print(f"Iteration: {k}, f(x): {func_val}, ||gradf||: {gradient}, alpha: {alpha_k}")
    
        if gradient <= stopping_point: 
            print("Final value of objective function: ", func_val)
            print("CONVERGED\n")
            return func(cur_vector), k
        
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
            alpha_k = Armijo_backtracking(cur_vector, func, gradient, p, 1)
        else: 
            alpha_k = Wolfe_linesearch(cur_vector, p, func, gradient_func)


        cur_vector = cur_vector + (alpha_k * p)
        k += 1


    print("Final value of objective function: ", func_val)
    print("HIT MAX ITERATIONS IN NEWTON CG")

    return func_val, k

def check_convergence(k): 
    if k >= K_MAX: 
        return "MAX ITERATIONS"
    else:
        return "CONVERGED"
    

if __name__ == "__main__":
    
    sys.stdout = open('output.txt', 'wt')
    pdf = FPDF(format='letter')
    pdf.add_page()
    pdf.set_font('Arial', size=8)
    
    one_v = [-1.2, 1]
    two_v = [-1] * 10
    three_v = [2] * 10
    four_v = [-1] * 100
    five_v = [2] * 100
    six_v = [2] * 1000
    seven_v = [2] * 10000
    eight_v = [1, 1]
    nine_v = [0, 0]
    ten_v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    vectors = [one_v, two_v, three_v, four_v, five_v, six_v, seven_v, eight_v, nine_v, ten_v]

    

    










