import numpy as np

# Initialize Q
Q = np.zeros(12)

# Set desired segment sum
segment_sum = 2

for i in range(1):
    valid = False
    while not valid:
        # Generate 12 random positive numbers
        temp = np.abs(np.random.rand(12))
        
        # Normalization step to ensure sum is 2
        temp = temp * segment_sum / np.sum(temp)
        
        # Check condition Q = Q*2/sum(Q)
        if np.abs(np.sum(temp * 2 / np.sum(temp)) - np.sum(temp)) < 1e-6:
            valid = True

    Q[i*12:(i+1)*12] = temp

print(Q)