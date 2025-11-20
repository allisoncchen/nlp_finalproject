### Details
Homework3_Programming Part
Allison Chen 
acc5329

NOTE
I've left in two calls to my Newton's and Modified Newton's method with 
question nine after my correction from HW2. I came to your office hours and we
reviewed the code after I corrected it a few weeks back and everything looked good.
### Files
* [runAssignment3.py] (hw3/runAssignment3.py)
Main file of the submission. Contains the code to run each algorithm, the
functions/gradients/hessian calculations, and the inputs. 

Rosenbrock Functions
def Rosenbrock_function(x_vector)
def Rosenbrock_gradient(x_vector)
def Rosenbrock_hessian(x_vector)

Beale Functions
def Beale_function(x_vector)
def Beale_gradient(x_vector)
def Beale_hessian(x_vector)

Unnamed Function (Problem 10)
def unnamed_function(x_vector)
def unnamed_gradient(x_vector)
def unnamed_hessian(x_vector)

Algorithm Functions Added for HW 2
def Armijo_backtracking(x, func, gradient, p, alpha_init)
def steepest_descent(x, func, gradient)
def newtons_method(x, func, gradient, hessian)
def cholesky_with_added_multiple_of_identity(A)
def modified_newtons_method(x, func, gradient, hessian)

Algorithm Functions Added for HW 3
def Wolfe_linesearch(x, p, func, gradient)
def BFGS(x, func, gradient)
def two_loop_recursion(cur_gradient, y_s_pairs, gamma, I): 
def L_BFGS_large_memory(x, func, gradient): 
def L_BFGS(x, func, gradient)
def Newton_CG(x, func, gradient_func, hessian): 

* [table.pdf] (hw3/table.pdf)
NOT YET GENERATED. Produced from running runAssignment3.py. Includes table and discussion for coding problems. 

* [requirements.txt] (hw3/requirements.txt)
Requirements file for all the libraries needed to run code. Output of pip freeze > requirements.txt. 

* [output.txt] (hw3/output.txt)
NOT YET GENERATED. Output generated from executing the main function in runAssignment2.py. For each question, includes statements for each iteration of an algorithm. 
