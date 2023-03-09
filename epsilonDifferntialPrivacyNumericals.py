import numpy as np

def query(data, epsilon):
    # Compute the sum of the data
    data_sum = np.sum(data)
    # Add Laplace noise to the sum
    noise = np.random.laplace(scale=1.0 / epsilon)
    noisy_sum = data_sum + noise
    # Return the noisy sum
    return noisy_sum


ages = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
print(query(ages, 0.1))
print(query(25, 0.1))