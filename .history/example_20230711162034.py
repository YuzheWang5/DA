import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = pd.read_csv('Q.csv')
data.head()
plt.plot(data)
plt.title('Q matrix')
plt.xlabel('time')
plt.ylabel('probility')
plt.show()