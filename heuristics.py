import numpy as np
import math

MAX_NUM = int(1e12)
# MAX_NUM = 10

debug = False

def random_algo(A, num_iterations = 25000):
    n = len(A)

    partitions = np.random.choice([-1,1], size=(num_iterations, n)) # each row is a partition

    values = partitions * A

    residues = np.abs(values.sum(axis=1)) # take row sums, and absolute values

    return residues.min()

def hill_climb_algo(A, num_iterations=25000):
    pass

def partition(A):
    n = len(A)
    indices = np.random.randint(0, n, size=n)
    A_prime = np.array([A[np.where(indices == i)[0]].sum() for i in range(n)])
    if debug:
        print(f'A: {A}, indices: {indices}, A_prime: {A_prime}')
    return A_prime

def run_experiments(algo, numbers = 5, trials=50, num_iterations=25000, prepartition = False):
    results = []
    for trial in range(trials):
        A = np.random.randint(0, MAX_NUM, size=numbers)
        if prepartition:
            P = partition(A)
        else:
            result = algo(A, num_iterations=num_iterations)
            results.append(result)
    return np.array(results)


run_experiments(random_algo)