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

    minimum_r = 1e15
    for iter in range(num_iterations):
        if partition:
            # initialize a random partition
            S = np.random.randint(0, n, size=n)
            A_prime = np.array([A[np.where(S == i)[0]].sum() for i in range(n)])
            minimum_r = min(minimum_r, kk(A_prime))
        else:
            # initialize a random assignment
            S = np.random.choice([-1, 1], size=n)
            minimum_r = min(minimum_r, np.abs(np.sum(S*A)))

    return minimum_r

def hill_climb_algo(A, num_iterations=25000, partition = False):
    n = len(A)
    # initialize some random solution
    S = None
    minimum_r = 0
    if partition:
        # initialize a random partition
        S = np.random.randint(0, n, size=n)
        A_prime = np.array([A[np.where(S == i)[0]].sum() for i in range(n)])
        minimum_r = kk(A_prime)
    else:
        # initialize a random assignment
        S = np.random.choice([-1, 1], size=n)
        minimum_r = np.abs(np.sum(S*A))

    for iter in range(num_iterations):
        S_prime = random_neighbor(S, partition=partition)
        r = 0 # residue
        if partition:
            A_prime = np.array([A[np.where(S_prime == i)[0]].sum() for i in range(n)])
            r = kk(A_prime)
        else:
            r = np.abs(np.sum(A*S_prime))

        # if it is a better point, we move
        if r < minimum_r:
            minimum_r = r
            S = S_prime
    
    # we could also return S if we wanted to
    return minimum_r

# return a random neighbor of a given set A or partition P
def random_neighbor(S, partition = False):
    n = len(S)
    if partition:
        # choose random i, j such that S[i] != j
        # set S[i] = j
        i, j = np.random.choice(list(range(0, n)), size=2, replace=False)
        while j != S[i]:
            i, j = np.random.choice(list(range(0, n)), size=2, replace=False)
        S[i]=j
        return S
    else:
        # choose random indices i, j
        # swap sign of i
        # randomly swap sign of j (with p=0.5)
        i, j = np.random.choice(list(range(0, n)), size=2, replace=False)
        S[i] *= -1
        to_swap = np.random.choice([-1,1])
        S[j] *= (-1 * (to_swap))
        return S

# given an algorithm, run that algorithm on the list of A_s given (A_list)
def run_experiments(algo, A_list, num_iterations=25000, prepartition = False):
    results = []
    for A in A_list:        
        result = algo(A, num_iterations=num_iterations, partition = prepartition)
        results.append(result)
        
    return np.array(results)

A_list = np.random.randint(0, MAX_NUM, size=(trials, numbers))

print(run_experiments(random_algo, A_list))
print(run_experiments(hill_climb_algo, A_list))
# print(run_experiments(random_algo, A_list, prepartition=True))
# print(run_experiments(hill_climb_algo, A_list, prepartition=True))