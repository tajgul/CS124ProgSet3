import numpy as np
import heapq
import math
import sys
import time 

def master(flag, code, input):

    MAX_NUM = int(1e12) # maximum that a_i can be
    trials = 50 # number of trials 
    numbers = 100 # number of elements in A
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
                A_prime = np.zeros(n)
                for i in range(n):
                    A_prime[S[i]] += A[i]
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
            A_prime = np.zeros(n)
            for i in range(n):
                A_prime[S[i]] += A[i]
            minimum_r = kk(A_prime)
        else:
            # initialize a random assignment
            S = np.random.choice([-1, 1], size=n)
            minimum_r = np.abs(np.sum(S*A))
        avg_time = 0
        for iter in range(num_iterations):
            start = time.time()
            S_prime = random_neighbor(S, partition=partition)
            r = 0 # residue
            if partition:
                A_prime = np.zeros(n)
                for i in range(n):
                    A_prime[S[i]] += A[i]
                r = kk(A_prime)
            else:
                r = np.abs(np.sum(A*S_prime))

            # if it is a better point, we move
            if r < minimum_r:
                minimum_r = r
                S = S_prime
            end = time.time()
            avg_time += (end - start)
        print(avg_time / num_iterations)
        print(avg_time)
        # we could also return S if we wanted to
        return minimum_r

    def cooling(n):
        return pow(10,10) * pow(0.8, n / 300)

    def sim_anneal(A, num_iterations = 25000, partition=False):
        n = len(A)
        rand_sol = None
        opt = None
        if partition: 
            S = np.random.randint(0, n, size=n)
            rand_sol = np.zeros(n)
            for i in range(n):
                rand_sol[S[i]] += A[i]
            opt = kk(rand_sol)
        else: 
            rand_sol = np.random.choice([-1,1], size=(1, n)).reshape(n) * A
            opt = sum(abs(rand_sol))

        true_opt = opt
        for iter in range(1, (num_iterations+1)):
            neighbor = random_neighbor(rand_sol, partition=partition)
            res_n = kk(neighbor) if partition else np.abs(np.sum(neighbor))
            if res_n < opt or (np.random.random() < math.exp(- (res_n - opt) / cooling(iter) )):
                rand_sol = neighbor
                opt = res_n
            true_opt = min(true_opt, opt)
        return true_opt

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
        
    part = code > 9
    if (code == 0):
        return kk(input)
    elif (code  == 1 or code == 11):
        return random_algo(input, partition=part)
    elif(code == 2 or code == 12):
        return hill_climb_algo(input, partition=part)
    elif(code == 3 or code == 13):
        return sim_anneal(input, partition=part)
    
if len(sys.argv) == 4:
    flag = sys.argv[1]
    code = int(sys.argv[2])
    input = sys.argv[3]
else:
    assert 2 == 3
A = []
#Unpack Input
with open(input, "r") as file:
    for line in file:
        A.append(int(line))

print("read")
output = master(flag, int(code), np.array(A))
print(output)