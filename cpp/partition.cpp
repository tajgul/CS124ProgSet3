#include <iostream>
#include <vector>
#include <string>
#include <ostream>
#include <queue>
#include <random>
#include <chrono>
#include <fstream>
#include <cstring>

using namespace std;

// experiment parameters
int NUM_ITERATIONS = 50000;
int NUM_TRIALS = 50;
int NUM_ELEMENTS = 100;

// important definitions
typedef long int li;
mt19937 rng(124); // Changed from set seed 124 
uniform_int_distribution<long int> gen(0, 1e12); // uniform, unbiased

// implementation of kk algorithm
long int kk(vector<li> A) {
    // max heap
    priority_queue<li> q; 
    for (auto num : A) {
        q.push( num);
    }
    while (q.size() > 1) {
        li first = q.top(); q.pop();
        li second = q.top(); q.pop();
        q.push(first-second);
    }
    return q.top();
}

// helper functions

// find the residual given A, P, and whethr we are using partition
// if partition is false, then P is an array of -1, 1
// otherwise P[i] is the partition of element i
inline long int find_residual(vector<long int> A, vector<long int> P, bool partition = false) {
    int n = A.size();
    if (partition) {
        vector<li> A_prime(n, 0);
        for (int i = 0; i < n; i++) {
            A_prime[P[i]] += A[i];
        }
        return kk(A_prime);
    } else {
        long int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[i] * P[i];
        }
        return abs(sum);
    }
}

// generate a random neighbor
// if partition = true, treat A as a partition
inline vector<long int> random_neighbor(vector<long int> A, bool partition= false) {
    int n = A.size();
    uniform_int_distribution<int> gen_n(0, n-1);
    uniform_int_distribution<int> gen_bin(0, 1);
    if (partition) {
        int i = gen_n(rng);
        int j = gen_n(rng);
        while (j == A[i]) j = gen_n(rng); // sample a different j
        A[i]=j; // Does nothing?
        return A;

    } else {
        int i = gen_n(rng);
        int j = i;
        while (j == i) j = gen_n(rng); // sample a different j
        A[i]=-A[i];
        if (gen_bin(rng) == 1) {
            A[j] = -A[j];
        } 
        return A;
    }
}

// generate random initialization
inline vector<long int> random_init(int n, bool partition = false){
    uniform_int_distribution<li> gen_n(0, n-1);
    uniform_int_distribution<li> gen_bin(0, 1);

    vector<li> S;
    if (partition) {
        for (int i = 0; i < n; i++) {
            S.push_back(gen_n(rng));
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (gen_bin(rng) == 1){
                S.push_back(1);
            } else {
                S.push_back(-1);
            }
        }
    }
    return S;
}

// random algo
long int random_algo(vector<long int> A, bool partition = false) {
    li n = A.size();
    vector<long int> S = random_init(n, partition);
    long int minimum_r = find_residual(A, S, partition);

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        vector<long int> S = random_init(n, partition);
        long int r = find_residual(A, S, partition);
        minimum_r = min(r, minimum_r);
    }

    return minimum_r;
}

// hill climbing 
long int hill_climbing(vector<long int> A, bool partition = false) {
    li n = A.size();
    
    // initialize S randomly, and find the residual
    vector<long int> S = random_init(n, partition);
    long int minimum_r = find_residual(A, S, partition);

    // generate random neighbors of S, move if the neighbor has better value
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        vector<li> S_prime = random_neighbor(S, partition);
        long int r = find_residual(A, S_prime, partition);
        if (r < minimum_r) {
            minimum_r = r;
            S = S_prime;
        }
    }

    return minimum_r;
}

// Auxillery function for simulated annealing
double cooling(int iter) {
    return pow(10, 10) * pow(0.8, floor(iter/300));
}

long int simulated_anneal(vector<long int> A, bool partition = false){
    li n = A.size(); 

    vector<long int> S = random_init(n, partition); 
    long int minimum_r = find_residual(A, S, partition);
    long int true_min_r = minimum_r;

    for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {
        vector<li> S_prime = random_neighbor(S, partition);
        long int r = find_residual(A, S_prime, partition);
        true_min_r = min(true_min_r, r);
        if (r < minimum_r || exp( (r - minimum_r) / cooling(iter))) {
            minimum_r = r;
            S = S_prime;
        }
    }
    return true_min_r;
}

int main(int argc, char** argv) {
    if (argc != 4){
        cout << "Error" << endl;
        return 0;
    }
    if (strcmp(argv[1],"1") == 0){  // If running for write up
        ofstream input_file(argv[3]);
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            // auto start = chrono::high_resolution_clock::now(); 
            vector<long int> A;
            for (int i = 0; i < NUM_ELEMENTS; i++){
                A.push_back(gen(rng));
            }
            // auto end = chrono::high_resolution_clock::now();
            // auto duration = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
            input_file << simulated_anneal(A, true)  << "\n";
        } 
        input_file.close();
    }
    else{ //If submitting on gradescope
        ifstream input_file(argv[3]);
        vector<li> A;
        std::string line;
        while (std::getline(input_file, line)) {
            A.push_back(stol(line) );
        }
        input_file.close();
        int arg = stoi(argv[2]);
        bool part = arg > 9;
        if (arg == 0){
            cout << kk(A) ;
        }
        else if(arg == 1 || arg == 11){
            cout << random_algo(A, part);
        }
        else if(arg == 2 || arg == 12){
            cout << hill_climbing(A, part);
        }
        else{
            cout<<  simulated_anneal(A, part);
        }
    }
    return 0;
}