import matplotlib.pyplot as plt
import sklearn.tree
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def func1(t):
    if t == 'Iris-setosa' :
        return int(0)
    elif t == 'Iris-versicolor':
        return int(1)
    elif t =='Iris-virginica':
        return int(2)

iris = pd.read_csv('iris.data',header=None)
col_name =   ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris.columns = col_name

iris2 = iris.copy()
iris2['species']=iris2['species'].apply(func1)
# sns.scatterplot(data=iris2, x="sepal_length", y="sepal_width",hue='species')
features3 = iris2.iloc[:, :2].values
labels3 = iris2.iloc[:, -1].values

# 决策树
dt2 = DecisionTreeClassifier(random_state=42)
dt2.fit(features3,labels3)
x_min, x_max = features3[:, 0].min() - 0.5, features3[:, 0].max() + 0.5
y_min, y_max = features3[:, 1].min() - 0.5, features3[:, 1].max() + 0.5
#绘制棋盘
h = 0.02  
plt.figure(figsize=(8,8))
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#对棋盘中的每一个交点进行分类预测
Z = dt2.predict(np.c_[xx.ravel(), yy.ravel()])
#把预测的结果展现在二维图像上，绘制决策边界
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#添加原始数据
plt.scatter(features3[:, 0], features3[:, 1], c=labels3, edgecolors="k", cmap=plt.cm.Paired)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
#坐标刻度和取值范围设置
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
#显示图像
plt.show()


# 感知机
clf2 = Perceptron()
clf2.fit(features3,labels3)
b = clf2.score(features3,labels3)
x_min, x_max = features3[:, 0].min() - 0.5, features3[:, 0].max() + 0.5
y_min, y_max = features3[:, 1].min() - 0.5, features3[:, 1].max() + 0.5
h=0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
xx.ravel()
Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(6, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.figure(1, figsize=(6, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(features3[:, 0], features3[:, 1], c=labels3, edgecolors="k", cmap=plt.cm.Paired)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()





# plt.figure(figsize=(18,12))
# _=sklearn.tree.plot_tree(clf,filled = True)
plt.show()



# model3 = LogisticRegression()
# model3.fit(features3,labels3)
# x_min, x_max = features3[:, 0].min() - 0.5, features3[:, 0].max() + 0.5
# y_min, y_max = features3[:, 1].min() - 0.5, features3[:, 1].max() + 0.5
# #绘制棋盘
# h = 0.02  
# plt.figure(figsize=(8,8))
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# #对棋盘中的每一个交点进行分类预测
# Z = model3.predict(np.c_[xx.ravel(), yy.ravel()])
# #把预测的结果展现在二维图像上，绘制决策边界
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# #添加原始数据
# plt.scatter(features3[:, 0], features3[:, 1], c=labels3, edgecolors="k", cmap=plt.cm.Paired)
# plt.xlabel("Sepal length")
# plt.ylabel("Sepal width")
# #坐标刻度和取值范围设置
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
# #显示图像
# plt.show()
