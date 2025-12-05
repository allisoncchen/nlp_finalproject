# ORI 391/CSE 393 Final Project 

This is the code package for the Final Project Submission to ORI 391/CSE 393 created by 
Parul Singh and Allison Chen.

December 2025.

### Install packages
```bash
pip install -r requirements.txt
```

### Package Structure

* algorithm_v_problem: FOLDER of plots to compare problem vs. iteration and
gradient vs. iteration computations for the run of each algorithm on each – 
generated from the code in [generate_big_table.py](scripts/generate_big_table.py).

    _Corresponds with section 3 of our NLP Final Report._

* parameter_search_results: FOLDER of plots used to answer our research question
(**What are the best linesearch parameters?**) – generated from the code in 
[create_all_tables.py](scripts/create_all_tables.py) and 
[main.py](scripts/main.py).

    _Corresponds with section 4 of our NLP Final Report._

* scripts: FOLDER of scripts that we used to either run our coded algorithms 
on the 12 problems OR investigate our research question.
    * [create_all_tables.py](scripts/create_all_tables.py): Creates multiple plots to compare each algorithm. 
    * [generate_big_table.py](scripts/generate_big_table.py): Creates one chart of the output of running every algorithm on every problem. 
    * [linesearch_testing.py](scripts/linesearch_testing.py): Code to investigate our research question. 
    * [main.py](scripts/main.py): Code to investigate our research question. 
    * [problems.py](scripts/problems.py): Function, gradient, and hessian implementation of each of the 12 problems.


* [nablaninjas.py](nablaninjas.py): Prompts user for input and will run the
specified algorithm and problem with user-decided parameters. 


