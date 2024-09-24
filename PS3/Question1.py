import numpy as np
import time
print('Using for loops')

def matrix_multiply(N):
    """Function to multiply two NxN random matrices with for loop and return the time taken."""
    A = np.random.randint(10, size=(N, N)) 
    B = np.random.randint(10, size=(N, N)) 
    C = np.zeros([N, N], float)
    
    start = time.time()
    for i in range(N): 
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    end = time.time()
    
    return end - start

# List of matrix sizes
N_values = [10, 20, 40, 80]
times = []

# Loop through different matrix sizes
for N in N_values:
    time_taken = matrix_multiply(N)
    times.append(time_taken)
    print(f'Time taken for N = {N}: {time_taken}')

# Calculate the scaling factor relative to the time for N = 10
factor = np.cbrt(np.array(times) / times[0])
print(f'Factor by which the time changes: {np.array(times)/times[0]} \ncube rooted the factors are: {factor}')
print("The time scalled with ~N^3")





# with dot()
print('Using for dot()')

def matrix_multiply_dot(N):
    """Function to multiply two NxN random matrices with dot() and return the time taken."""
    A = np.random.randint(10, size=(N, N)) 
    B = np.random.randint(10, size=(N, N)) 
    C = np.zeros([N, N], float)
    
    start = time.time()
    C = np.dot(A,B)
    end = time.time()
    
    return end - start


# List of matrix sizes
N_values = [10, 20, 40, 80]
times_dot = []

# Loop through different matrix sizes
for N in N_values:
    time_taken_dot = matrix_multiply_dot(N)
    times_dot.append(time_taken_dot)
    print(f'Time taken for N with dot() = {N}: {time_taken_dot}')

# Calculate the scaling factor relative to the time for N = 10
factor_dot = np.cbrt(np.array(times_dot) / times_dot[0])
print(f'Factor by which the time changes: {np.array(times_dot)/times_dot[0]} \ncube rooted the factors are: {factor_dot}')
print("The time taken was much lower than using for loops")

