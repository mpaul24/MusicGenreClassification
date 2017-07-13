import pandas as pd
import numpy as np
dataset = pd.read_csv('Accuracy_k5.csv', header=None)
dataset = dataset.values
print(np.mean(dataset))