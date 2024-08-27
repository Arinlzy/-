import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

iris_dataset = load_iris()
iris_df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
iris_df['species'] = iris_dataset.target_names[iris_dataset.target]

# 创建一个 2x2 的子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 循环遍历每个子图的索引和对应的特征组合，并绘制散点图
for i, (x, y) in enumerate([[0, 1], [0, 2], [1, 3], [2, 3]]):
    sns.scatterplot(x=iris_df.iloc[:, x], y=iris_df.iloc[:, y], hue=iris_df['species'], palette='Set1', ax=axs[i // 2, i % 2])
    axs[i // 2, i % 2].set_xlabel(iris_dataset.feature_names[x])
    axs[i // 2, i % 2].set_ylabel(iris_dataset.feature_names[y])
    axs[i // 2, i % 2].set_title(f" {iris_dataset.feature_names[x]} and {iris_dataset.feature_names[y]}")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i in list(range(0,4)):
    sns.violinplot(x=iris_df["species"], y=iris_df.iloc[:, i], ax= axs[i // 2, i % 2])
plt.show()