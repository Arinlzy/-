from calendar import c
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris() #获取数据
df = pd.DataFrame(iris.data, columns=iris.feature_names)# 获取列的属性值
df['label'] = iris.target# 增加一个新列


df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label'] # 重命名各个列
df.label.value_counts() # 计算label列0、1、2出现的次数

"""
绘制散点图
"""
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.scatter(df[100:]['sepal length'], df[100:]['sepal width'], label='2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])# 按行索引,取前100行,取第0，1列以及最后1列

X, y = data[:,:-1], data[:,-1] #X: {ndarray:(100, 2)}  y: {ndarray:(100, )}

y = np.array([1 if i == 1 else -1 for i in y])# 将存在y中的数据为0的值改为-1


# 数据线性可分，二分类数据
# 此处为一元一次线性方程 
class Model:
   def __init__(self):# 初始化数据
       self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
       self.b = 0
       self.l_rate = 0.1 
       # self.data = data

   def sign(self, x, w, b):
       y = np.dot(x, w) + b
       return y

   # 随机梯度下降法
   def fit(self, X_train, y_train):
       is_wrong = False
       while not is_wrong:
           wrong_count = 0# 初始设置错误次数为0
           for d in range(len(X_train)):
               X = X_train[d]
               y = y_train[d]
               if y * self.sign(X, self.w, self.b) <= 0:
#权重self.w和截距self.b更新
                   #TODO XXX
                   self.w = self.w + self.l_rate * X * y
                   #TODO XXX
                   self.b = self.b + self.l_rate * y
                   wrong_count += 1
           if wrong_count == 0:# 误分点数目为0跳出循环
               is_wrong = True
       return 'Perceptron Model!'

   def score(self):
       pass
    

perceptron = Model()
perceptron.fit(X, y)

x_points = np.linspace(4, 7, 10)
#计算输出值y
#TODO XXX
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()


from sklearn.linear_model import Perceptron# 使用scikit-learn自带的感知机模型

#配置导入的感知机模型clf
#TODO XXX
clf = Perceptron()
#使用上面的训练数据代入模型中进行训练
#TODO XXX
clf.fit(X, y)
# Weights assigned to the features.
print(clf.coef_)

# 截距 Constants in decision function.
print(clf.intercept_)

x_points = np.arange(4, 8)
#计算输出值y_
#TODO XXX
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]

plt.plot(x_points, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
