from pprint import pprint

import numpy as np
from pulp import *
import time
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def compute_profit(x, y):
    """
    Calculate features profit using Information Gain.

    :param x: array of features values
    :param y: array of features' classes

    :return: a dictionary containing the IG value for each feature
    """
    # Create mutual_info_classif object to calculate information gain values
    ig_scores = mutual_info_classif(x, y)
    return dict(zip(x.columns, ig_scores))


def create_knapsack_model(w, p, b):
    """
    Utility function that create the knapsack problem's model by setting variables, constraints ad objective function

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


def solve_knapsak(problem):
    """
    Solve the given knapsack problem using Coin-OR Branch and Cut

    :param problem: the problem to solve

    :return: list containing the items to be taken to maximize the profit
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

    # The optimised objective function value is printed to the screen
    # print("Total profit = ", value(problem.objective))

    # Some more info about the solution (only variables / items that are selected)
    '''
    used_cap = 0.0
    print("Used items:")
    for i in range(num_items):
        if x[i].value() == 1:
            print(i, items[i])
            used_cap += items[i][1]
    print("Profit: %d - Used capacity: %d (/ %d)" % (value(problem.objective), used_cap, b))
    '''
    return solution

# dataset = pd.read_csv('datasets/iris.data')
dataset = pd.read_csv('datasets/diabetes.csv')  # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# todo se i valori devono essere interi li arrotondo in qualche modo

print('\nDataset:\n', dataset)
profit = compute_profit(dataset[dataset.columns[0:(dataset.columns.size - 1)]],
                        dataset[dataset.columns[-1]])  # last column is for belonging class

print('\nProfit:')
pprint(profit)

# removing belonging class column
dataset.drop(dataset.columns.values[-1], axis=1, inplace=True)

# extraction of relevant data for the problem
features = pd.DataFrame(index=dataset.columns, columns=['x', 'w', 'p'])
features['x'] = np.zeros(dataset.columns.size)
features['w'] = dataset.mean()
features['p'] = profit


#######################################
###############  PLI 1  ###############
#######################################

# con questa b intendo che prendo il peso medio delle features e lo moltiplico per il numero di features che voglio
# avere, in modo da avere il valore che voglio conservare

# set b (knapsack capacity) as the average weight we would like to reach
reduct_to = 3  # approximate number of features we would like to reach
b = features['w'].mean() * reduct_to  # todo PROBLEMA: dovrei mettere i pesi tutti =1 e mettere b = #feature_a_cui_voglio_ridurre_il_problema

# create the model for the problem to solve
problem = create_knapsack_model(w=features['w'], p=features['p'], b=b)

# Print the model created for the problem
print('\n----------------------------------------------------')
print('\t\t\t\tProblem formulation')
print('----------------------------------------------------')
print(problem)
print('----------------------------------------------------\n')

# Write the model to disk
# problem.writeLP("Knapsack.lp")

# solve the problem using Pulp interface for Cbc
solution = solve_knapsak(problem)
features['x'] = solution

# print(b)
print('\nFinal result:')
print(features.to_string())


'''
FORSE USARE QUESTO PER SETTARE PESI E PROFITTI PUÒ ESSERE UNA SOLUZIONE

from sklearn.feature_selection import SelectKBest, f_classifbestfeatures = SelectKBest(score_func=f_classif, k=3)
iris_trim = bestfeatures.fit_transform(iris_norm, target)print(bestfeatures.scores_)
print(bestfeatures.pvalues_)
print(iris_trim.shape)
'''

#######################################
###############  PLI 2  ###############
#######################################
