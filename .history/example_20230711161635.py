import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = pd.read_csv('Q.csv')
data.head()
fig1 = plt.figure(figuresize=(8,5))
plt.plot(data)
plt.title('Q matrix')
plt.show()