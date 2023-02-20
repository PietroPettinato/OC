import tracemalloc
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

    # Add the objective function
    problem += lpSum([x[i] * (items[i])[1] for i in range(num_items)]), "Objective: Maximize profit"

    # Capacity constraint: the sum of the weights must be less than the capacity
    problem += lpSum([x[i] * (items[i])[0] for i in range(num_items)]) <= b, "Constraint: Max capacity"

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

    # Add the objective function
    problem += lpSum([x[i] for i in range(num_items)]), "Objective: Minimize features"

    # Treshold constraint: the sum of the profits must be greater than the treshold
    problem += lpSum([x[i] * profit[j] for i, j in zip(range(num_items), profits.keys())]) >= treshold, "Constraint: Profit treshold"

    return problem


def solve_problem(problem):
    """
    Solve the given problem using Coin-OR Branch and Cut and print some info

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


def run_performance_tests():
    def generate(row, cols, file_name):
        """
        Generate a dataset of size row x cols with random values in 0-1 range

        :param row: number of examples
        :param cols: list with columns' names
        :param file_name: name of the file to save the dataset

        :return: the dataset
        """
        dataset = pd.DataFrame(np.random.rand(row, len(cols)), columns=cols)
        # dataset = pd.DataFrame(np.random.randint(1, 50, (row, len(cols))), columns=cols)
        dataset['Class'] = np.random.randint(0, 2, size=row)
        # print(file_name, '\n\n', dataset.to_string(), end='\n\n\n')
        # dataset.to_csv(f'datasets/{file_name}.csv', index=False)
        return dataset

    def solve_problem_short(problem):
        """
        Solve the given problem using Coin-OR Branch and Cut and print some info

        :param problem: the problem to solve

        :return: list containing the items to be taken
        """
        # Solve the optimization problem
        tracemalloc.start()
        start_time = time.time()
        problem.solve()
        print(f"||| Solved in {time.time() - start_time} seconds.")
        print(f"||| Memory used:  {(tracemalloc.get_traced_memory()[1] / 10 ** 3) / 1024: .2f} MB")
        tracemalloc.stop()
        # Was the problem solved to optimality?
        print("||| Status:", LpStatus[problem.status], end='\n')
        # return solution

    chars = list('ABCDEFGHILMNOPQRSTUVZ1234567890JKWXY')
    cols = ''

    for c in chars:
        '''
        if c not in list('ABCDEFGHILMNOPQRSTUVZ'):
            continua = input('continuare? (n per uscire)')
            if continua == 'n':
                print('Stopped')
                break
        '''
        cols += c
        # print('features:', cols)

        print(f'\n\n\t ||| Esecuzione con {len(list(cols))} features')
        dataset = generate(10, list(cols), 'soccer')

        data = dataset[dataset.columns[0:(dataset.columns.size - 1)]]
        belonging_class = dataset[dataset.columns[-1]]  # last column is for belonging class
        data = data.astype(float)  # convert values to float
        profits = compute_profit(data, belonging_class)
        # items = [(a, b) for a, b in zip(data.mean(), profits.values())]

        features = data.mean()
        reduct_to = len(features) * (3 / 4)
        b = features.mean() * reduct_to

        tracemalloc.start()
        problem = create_knapsack_model(w=data.mean(), p=profits.values(), b=b)
        print(f"||| Memory used MODEL:  {(tracemalloc.get_traced_memory()[1] / 10 ** 3) / 1024: .2f} MB")
        tracemalloc.stop()
        solve_problem_short(problem)


# uncomment to run tests on randomly generated dataset with increased number of features
# run_performance_tests()
# exit()

select = int(input("choose the dataset:\n 0 : iris\n 1 : diabetes\n 2 : bankrupt\n> "))
if select == 0:
    dataset = pd.read_csv('datasets/iris.data')
    data = dataset[dataset.columns[0:(dataset.columns.size - 1)]]
    belonging_class = dataset[dataset.columns[-1]]  # last column is for belonging class
elif select == 1:
    dataset = pd.read_csv('datasets/diabetes.csv')  # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    data = dataset[dataset.columns[0:(dataset.columns.size - 1)]]
    belonging_class = dataset[dataset.columns[-1]]  # last column is for belonging class
else:
    dataset = pd.read_csv('datasets/bankrupt.csv')  # https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction
    data = dataset[dataset.columns[1:dataset.columns.size]]
    belonging_class = dataset[dataset.columns[0]]  # first column is for belonging class

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

# set b (knapsack capacity) as the average weight we would like to reach
reduct_to = len(features) * (3 / 4)  # approximate number of features we would like to reach, we want to have 3/4 of the original set
b = features['w'].mean() * reduct_to

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
treshold = features['p'].sum() * (2 / 3)  # we want to keep at list 2/3 of total profit

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

