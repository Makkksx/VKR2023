import numpy as np
import pandas as pd
import time


def ffa_algorithm(table, costs, pop_size=30, max_iter=150, gamma=1.0, betta_0=1.0, notation="CS", transfer_fun="stan",
                  distance="euclid", betta_pow=2, alpha=0.5, alpha_inf=None, alpha_0=None,
                  gamma_alter=0, move_type=None):
    discrete = standard_discrete
    binary_fun = binarization(get_transfer_function(transfer_fun), discrete)
    repair_fun = repair_solution(table, costs, notation)
    fireflies = generate_solution((pop_size, len(costs)))
    curr_best = np.ones(len(costs))
    curr_best_intensity = np.inf

    if gamma_alter > 0:
        gamma = gamma / vector_dist(np.ones(len(costs)), np.zeros(len(costs)), distance) ** gamma_alter
        print(f"Gamma: {gamma}")

    get_attractive = calc_attractive

    if alpha_0 == 0 or alpha_inf == 0:
        get_alpha = None
    else:
        get_alpha = lambda t: alpha_inf + (alpha_0 - alpha_inf) * (np.e ** -t)

    def lambda_move_best(x1, x2, betta, alpha=0.1):
        U = np.random.uniform(-1, 1, x1.shape)
        return x1 + betta * (x2 - x1) + alpha * U * (x1 - curr_best)

    def lambda_move(x1, x2, betta, alpha=0.1):
        U = np.random.uniform(-1, 1, x1.shape)
        return x1 + betta * (x2 - x1) + alpha * U

    if move_type == 1:  # standart
        move_fun = move_fireflies
    elif move_type == 2:  # lambda_best
        move_fun = lambda_move_best
    elif move_type == 3:  # lambda
        move_fun = lambda_move
    else:
        raise ValueError("Error")

    _ = list(map(repair_fun, fireflies))

    light_intensity = calc_fitness(fireflies, costs)

    for step in range(max_iter):
        print(step)
        print(f"Elapsed time {time.time() - start_time}")
        for i in range(len(fireflies)):
            for j in range(0, i):
                if light_intensity[j] < light_intensity[i]:
                    fireflies[i] = move_fun(fireflies[i], fireflies[j],
                                            get_attractive(betta_0, gamma,
                                                           vector_dist(fireflies[i], fireflies[j], distance),
                                                           betta_pow), alpha)
                    fireflies[i] = binary_fun(fireflies[i])
                    repair_fun(fireflies[i])
                    light_intensity[i] = calc_fitness(fireflies[i], costs)

        if get_alpha:
            alpha = get_alpha(step)

        best = np.argmin(light_intensity)
        if curr_best_intensity > light_intensity[best]:
            print(light_intensity[best])
            curr_best_intensity = light_intensity[best]
            curr_best = fireflies[best].copy()

    return curr_best, curr_best_intensity


def move_fireflies(x1, x2, betta, alpha=0.1):
    rand = np.random.sample(len(x1))
    return x1 + betta * (x2 - x1) + alpha * (rand - 0.5)


def calc_attractive(betta, gamma, r, m=2):
    return betta * np.e ** (-gamma * (r ** m))


def vector_dist(x1, x2, norm_type="euclid"):
    if norm_type.lower() == "euclid":
        ord = 2
    elif norm_type.lower() == "manhattan":
        ord = 1
    elif norm_type.lower() == "chebyshev" or norm_type.lower() == "cheb":
        ord = np.Inf
    else:
        raise Exception(f"Incorrect value {norm_type} for parameter type!")
    return np.linalg.norm(x2 - x1, ord=ord)


def generate_solution(size):
    return np.random.randint(0, 2, size).astype("float")


def calc_fitness(solution, costs):
    dim = 0 if solution.ndim == 1 else 1
    return np.array(np.sum(np.multiply(solution, costs), dim), dtype="float32")


def get_transfer_function(transfer_fun="s1"):
    if transfer_fun == 1:
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-2.0 * x))
    elif transfer_fun == 2:
        transfer = lambda x: 1.0 / (1.0 + np.e ** -x)
    elif transfer_fun == 3:
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 2.0))
    elif transfer_fun == 4:
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 3.0))
    elif transfer_fun == 5:
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-3.0 * x))
    elif transfer_fun == 6:  # stan
        transfer = lambda x: np.abs(2 / np.pi * np.arctan(x * np.pi / 2))
    else:
        raise Exception(f"Incorrect value {transfer_fun} for parameter transfer_fun!")
    return transfer


def standard_discrete(transfer_fun, x):
    x = transfer_fun(x)
    r = np.random.sample(x.shape)
    return np.where(x >= r, 1.0, 0.0)


def binarization(transfer, discretization):
    def decorator(x):
        if x.ndim == 1:
            return discretization(transfer, x)
        else:
            return np.array([discretization(transfer, e) for e in x], dtype=float)

    return decorator


def repair_solution(table, costs, notation):
    rows = {elem[0] for elem in table}
    columns = {elem[1] for elem in table}
    alpha = {f: s for f, s in zip(
        rows, [{elem[1] for elem in table if elem[0] == t} for t in rows]
    )}
    betta = {f: s for f, s in zip(
        columns, [{elem[0] for elem in table if elem[1] == t} for t in columns]
    )}

    if notation.lower() == "sc":
        alpha, betta = betta, alpha
        rows = columns

    def wrapped(solution):
        S = {i + 1 for i, e in enumerate(solution) if e == 1.0}
        w_num = {e1: e2 for e1, e2 in zip(
            rows, [len(S & alpha[i]) for i in rows]
        )}
        U = {e for e in w_num if w_num[e] == 0}
        while U:
            row = U.pop()
            j = min(alpha[row],
                    key=lambda r: np.Inf if len(U & betta[r]) == 0
                    else costs[r - 1] / len(U & betta[r]))
            S.add(j)
            for curr in betta[j]: w_num[curr] += 1
            U = U - betta[j]

        S = list(reversed(list(S)))
        for row in S[:]:
            for curr in betta[row]:
                if w_num[curr] < 2:
                    break
            else:
                S.remove(row)
                for c in betta[row]: w_num[c] -= 1
        S = np.array(S) - 1
        solution[:] = np.zeros(len(solution))
        solution[S] = 1.0

    return wrapped


def check_decision(data, result):
    n_cols = len(set(data.j))
    if n_cols == len(set(data[data.i.isin(result)].j)):
        print("Decision is correct")
    else:
        print("DECISION WRONG")



start_time = time.time()

# set params
notation = "sc"
transfer_fun = 1
event_horizon = "euclid"

pop_size = 160
max_iter = 100
InputDataSet = pd.read_csv("gen_test_160x1600.csv", header=None, names=['i', 'j'])
InputDataSet['Cost'] = 1
if notation.lower() == "sc":
    cover_rows = np.unique(InputDataSet["i"])
else:
    cover_rows = np.unique(InputDataSet["j"])

tmp_lst = InputDataSet["j"].tolist()
costs = np.array(InputDataSet["Cost"][[tmp_lst.index(e) for e in cover_rows]])
table = np.array(InputDataSet[["i", "j"]])


optimum, values = ffa_algorithm(table, costs, pop_size=pop_size, max_iter=max_iter, gamma=1, betta_0=1,
                                notation=notation,
                                transfer_fun=transfer_fun, distance=event_horizon, betta_pow=2,
                                alpha=1, alpha_inf=0, alpha_0=0, gamma_alter=0,
                                move_type=1)
print(f"Elapsed time {time.time() - start_time}")
print(f"Result cost: {values}")
print(f"Optimuum: {[i + 1 for i, e in enumerate(optimum) if e > 0.0]}")

optimum = pd.DataFrame(np.where(optimum > 0.0)[0] + 1, columns=["Row_"])
check_decision(InputDataSet, list(optimum.Row_))
