import numpy as np
import heapq
import math

MAX_NUM = int(1e12) # maximum that a_i can be
trials = 50 # number of trials 
numbers = 100 # number of elements in A
# MAX_NUM = 10

debug = False

def kk(A):
    h = [-i for i in A]  
    heapq.heapify(h)
    while len(h) > 1:
        heapq.heappush(h, -((-heapq.heappop(h))  - (-heapq.heappop(h))) )
    return -h[0] 

def random_algo(A, num_iterations = 25000, partition = False):
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

# return a random neighbor of a given set A or partition P
def random_neighbor(A, partition = False):
    if partition:
        raise NotImplementedError
    else:
        raise NotImplementedError

# given an algorithm, run that algorithm on the list of A_s given (A_list)
def run_experiments(algo, A_list, num_iterations=25000, prepartition = False):
    results = []
    for A in A_list:
        n = len(A)
        if prepartition:
            A = partition(A)
        
        result = algo(A, num_iterations=num_iterations, partition = prepartition)
        results.append(result)
        
    return np.array(results)

A_list = np.random.randint(0, MAX_NUM, size=(trials, numbers))

print(run_experiments(random_algo, A_list))