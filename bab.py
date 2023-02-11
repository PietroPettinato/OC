from pprint import pprint

import numpy as np
from pulp import *
import time
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def compute_profit(x, y):
    """
    Calculate features' profit using Information Gain.

    :param x: array of features values
    :param y: array of features' classes

    :return: a dictionary containing the IG value for each feature
    """
    # Create mutual_info_classif object to calculate information gain values
    ig_scores = mutual_info_classif(x, y)
    return dict(zip(x.columns, ig_scores))


def create_knapsack_model(w, p, b):
    """
    Function that create the knapsack problem's model by setting variables, constraints and objective function

    :param w: weights of the items
    :param p: profits of the items
    :param b: max knapsack capacity

    :return: the problem
    """
    # A list of tuples of items (profits, weight)
    items = [(a, b) for a, b in zip(w, p)]

    # number of items
    num_items = len(items)

    # Decision variables (array), x[i] gets 1 when item i is included in the solution
    x = pulp.LpVariable.dicts('item', range(num_items),
                              lowBound=0,
                              upBound=1,
                              cat='Integer')

    # Initialize the problem and specify the type
    problem = LpProblem("Knapsack_Problem", LpMaximize)

    # Capacity constraint: the sum of the weights must be less than the capacity
    problem += lpSum([x[i] * (items[i])[0] for i in range(num_items)]) <= b, "Constraint: Max capacity"

    # Add the objective function
    problem += lpSum([x[i] * (items[i])[1] for i in range(num_items)]), "Objective: Maximize profit"

    return problem


def create_problem_model(profit, treshold):
    """
    Function that create the problem's mathematical model by setting variables, constraints and objective function

    :param profit: profits of the items
    :param treshold: profit treshold to reach (P)

    :return: the problem
    """
    # number of profit
    num_items = len(profit)

    # Decision variables (array), x[i] gets 1 when item i is included in the solution
    x = pulp.LpVariable.dicts('item', range(num_items),
                              lowBound=0,
                              upBound=1,
                              cat='Integer')

    # Initialize the problem and specify the type
    problem = LpProblem("Feature_Problem", LpMinimize)

    # Capacity constraint: the sum of the weights must be less than the capacity
    problem += lpSum([x[i] * profit[j] for i, j in zip(range(num_items), profits.keys())]) >= treshold, "Constraint: Profit treshold"

    # Add the objective function
    problem += lpSum([x[i] for i in range(num_items)]), "Objective: Minimize features"

    return problem


def solve_problem(problem):
    """
    Solve the given problem using Coin-OR Branch and Cut

    :param problem: the problem to solve

    :return: list containing the items to be taken
    """
    # Solve the optimization problem
    start_time = time.time()
    problem.solve()
    print(f"Solved in {time.time() - start_time} seconds.")

    # Was the problem solved to optimality?
    print("Status:", LpStatus[problem.status], end='\n')

    # Take selected items and print them with it's resolved optimum value
    solution = []
    for v in problem.variables():
        solution.append(v.varValue)
        print(v.name, "=", v.varValue)

    return solution


select = int(input("choose the dataset:\n 0 : iris\n 1 : diabetes\n> "))
if select == 0:
    dataset = pd.read_csv('datasets/iris.data')
else:
    dataset = pd.read_csv('datasets/diabetes.csv')  # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

data = dataset[dataset.columns[0:(dataset.columns.size - 1)]]
belonging_class = dataset[dataset.columns[-1]]  # last column is for belonging class

data = data.astype(float)  # convert values to float

print('\nDataset:\n', dataset)
profits = compute_profit(data, belonging_class)

print('\nProfit:')
pprint(profits)

features = pd.DataFrame(index=data.columns, columns=['x', 'w', 'p'])
features['x'] = np.zeros(data.columns.size)
features['w'] = data.mean()
features['p'] = profits



#######################################
###############  PLI 1  ###############
#######################################

# con questa b intendo che prendo il peso medio delle features e lo moltiplico per il numero di features che voglio
# avere, in modo da avere il valore che voglio conservare

# set b (knapsack capacity) as the average weight we would like to reach
reduct_to = 3  # approximate number of features we would like to reach
b = features['w'].mean() * reduct_to
# b = features['w'].sum() * (2/3)

# create the model for the problem to solve
problem = create_knapsack_model(w=features['w'], p=features['p'], b=b)

# Print the model created for the problem
print('\n----------------------------------------------------')
print('\t\t\tProblem formulation PLI 1')
print('----------------------------------------------------')
print(problem)
print('----------------------------------------------------\n')

# Write the model to disk
# problem.writeLP("Knapsack.lp")

# solve the problem using Pulp interface for Cbc
solution = solve_problem(problem)
features['x'] = solution

print('\nFinal result PLI 1:')
print(features.to_string())
print('\nb value:  ', b, end='\n')
input("\n\nPLI 1 completed, press enter to continue with PLI 2...")


#######################################
###############  PLI 2  ###############
#######################################

# set profit treshold
treshold = features['p'].sum() * (2/3)  # we want to keep at list 2/3 of total profit

# create the model for the problem to solve
problem = create_problem_model(profits, treshold)

# Print the model created for the problem
print('\n----------------------------------------------------')
print('\t\t\tProblem formulation PLI 2')
print('----------------------------------------------------')
print(problem)
print('----------------------------------------------------\n')

# solve the problem using Pulp interface for Cbc
solution = solve_problem(problem)
features['x'] = solution

print('\nFinal result PLI 2:')
print(features.to_string())
print('\ntreshold value:  ', treshold, end='\n')

