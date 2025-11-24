
import argparse
def main(): 
    parser = argparse.ArgumentParser(description="A script to run problems for the NLP final project.")

    parser.add_argument("--problem_name", type=str, help="The problem you wish to run.")
    parser.add_argument("--x0", type=str, help="The starting point for the problem.")
    
    parser.add_argument("--method_name", type=str, help="The method to be used for solving the problem.")
    parser.add_argument("--options", nargs='+', help="Specify options in the order tol, max iterations, c1, c2, ")

    args = parser.parse_args()

    problem = args.problem_name
    x0 = args.x0
    method = args.method_name
    options = args.options

    print("You're about to run script to run problems for the NLP final project.")

    problem_name = input("Input the problem you wish to run:")
    x0 = input("Input the starting value you wish to use")
    method_name = input("Input the method name you wish to use")
    
    print(f"You're running {problem_name} with a starting value of {x0} using {method_name}.")
    



if __name__ == "__main__":
    main() 