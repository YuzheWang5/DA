import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = pd.read_csv("Q.csv", header=None)
Q = dataset
print Q