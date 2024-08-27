import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

iris_dataset = load_iris()
iris_df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
iris_df['species'] = iris_dataset.target_names[iris_dataset.target]

print(iris_df)
print(iris_df.iloc[:, 0:2])

fig ,axs = plt.subplots(3,3,figsize=(12,9))

plt.show()