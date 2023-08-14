import numpy as np

Q = np.zeros((36,1))
segment_sum = 2 # kb = 2
for i in range(3):
    valid = False
    while not valid:
        temp = np.abs(np.random.rand(12,1))
        temp = temp * segment_sum / np.sum(temp)
        if np.abs(np.sum(temp * 2 / np.sum(temp)) - np.sum(temp)) < 1e-6:
            valid = True
Q[i*12:(i+1)*12] = temp

np.savetxt("Q_matrix.csv", Q, delimiter=",")